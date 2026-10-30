"""Controlled experiment heads sharing the single clean foreground implementation."""
import torch
from torch import nn
from torch.nn import functional as F
from .progressive_foreground_refinement import ProgressiveForegroundRefinement

def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ForegroundOnly(nn.Module):
    def __init__(self, backbone_checkpoint=None):
        super().__init__()
        self.v3 = ProgressiveForegroundRefinement(backbone_checkpoint=backbone_checkpoint)

    def forward(self, image):
        base = self.v3(image)
        logit = base['final']
        p = logit.sigmoid()
        return {**base, 'base_final': logit, 'union': logit, 'P0': p, 'Pu': p}

class DirectReadout(ForegroundOnly):
    """H(PFG), H(S3,PFG), H(Fdec), or H(S3); foreground auxiliary loss retained."""
    def __init__(self, source, backbone_checkpoint=None):
        super().__init__(backbone_checkpoint)
        self.source = source
        channels = {'pfg_head': 1, 'semantic_pfg_head': 320,
                    'decoder_head': 192, 'semantic_head': 320}[source]
        self.input_projection = nn.Sequential(nn.Conv2d(channels,96,1,bias=False),
                                             nn.GroupNorm(8,96),nn.SiLU(inplace=True))
        self.class_head = nn.Sequential(
            nn.Conv2d(96+int(source=='semantic_pfg_head'),96,3,padding=1,bias=False),
            nn.GroupNorm(8,96), nn.SiLU(inplace=True), nn.Conv2d(96,3,1))
        self._features = {}
        if source == 'decoder_head':
            self.v3.boundary_block.smooth_body.register_forward_hook(self._smooth)
            self.v3.boundary_block.sharp_roi_body.register_forward_hook(self._sharp)

    def _smooth(self, module, inputs, output): self._features['smooth'] = output
    def _sharp(self, module, inputs, output): self._features['sharp_roi'] = output

    @staticmethod
    def restore_rois(features, boxes, size, batch_size):
        h,w = size
        canvas = features.new_zeros(batch_size,features.shape[1],h,w)
        for i,box in enumerate(boxes):
            b=int(box[0].item())
            x1=max(0,min(w-1,int(box[1].floor().item())))
            y1=max(0,min(h-1,int(box[2].floor().item())))
            x2=max(x1+1,min(w,int(box[3].ceil().item())))
            y2=max(y1+1,min(h,int(box[4].ceil().item())))
            canvas[b:b+1,:,y1:y2,x1:x2]=resize(features[i:i+1],(y2-y1,x2-x1))
        return canvas

    def forward(self, image):
        self._features.clear()
        try:
            base=super().forward(image)
            size=(image.shape[-2]//4,image.shape[-1]//4)
            if self.source=='pfg_head':
                features=self.input_projection(resize(base['P0'],size))
            elif self.source in ('semantic_pfg_head','semantic_head'):
                features=self.input_projection(resize(base['semantic_features'][2],size))
                if self.source=='semantic_pfg_head':
                    features=torch.cat((features,resize(base['P0'],size)),1)
            else:
                smooth=self._features['smooth']
                sharp=self.restore_rois(self._features['sharp_roi'],base['roi_boxes'],
                                        smooth.shape[-2:],image.shape[0])
                features=self.input_projection(resize(torch.cat((smooth,sharp),1),size))
            logits=resize(self.class_head(features),image.shape[-2:])
            return {**base,'logits':logits,'three_class_probability':logits.softmax(1)}
        finally:
            self._features.clear()
