"""One model registry; no imports from another checkout or implicit checkpoint."""
from config import CFG
from .foreground_type_allocation import SegDis
from .readouts import ForegroundOnly, DirectReadout

VARIANTS = ('segdis', 'segdis_no_correction', 'segdis_no_rta', 'flr', 'pfg_head',
            'semantic_pfg_head', 'decoder_head', 'semantic_head')

def build_model(backbone_checkpoint=None, backbone_name='pvt_v2_b2', variant='segdis'):
    if backbone_name != 'pvt_v2_b2':
        raise ValueError('Current experiments use the fixed PVT-v2-B2 encoder')
    if variant == 'segdis':
        model = SegDis(backbone_checkpoint, backbone_name)
    elif variant == 'segdis_no_correction':
        model = SegDis(backbone_checkpoint, backbone_name, correction_enabled=False)
    elif variant == 'segdis_no_rta':
        model = SegDis(backbone_checkpoint, backbone_name, correction_enabled=False,
                       relation_conditioning=False)
    elif variant == 'flr':
        model = ForegroundOnly(backbone_checkpoint)
    elif variant in VARIANTS:
        model = DirectReadout(variant, backbone_checkpoint)
    else:
        raise ValueError(f'Unknown model variant: {variant}')
    model.architecture = CFG.architecture
    model.variant = variant
    model.backbone_name = backbone_name
    return model

def three_class_probability(output):
    if 'three_class_probability' not in output:
        raise ValueError('A binary foreground model has no SOD/COD type probabilities')
    return output['three_class_probability']

def validate_checkpoint(checkpoint, model=None):
    from utils.checkpoint import validate_checkpoint as validate
    return validate(checkpoint, model)
