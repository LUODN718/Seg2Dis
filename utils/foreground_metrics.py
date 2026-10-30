"""Explicit raw-probability binary metrics; never infer SOD/COD types from them."""
import numpy as np

class ForegroundMetrics:
    def __init__(self,normalize=False):
        from py_sod_metrics import Smeasure,WeightedFmeasure,Fmeasure,Emeasure
        self.metrics={'sm':Smeasure(),'wfm':WeightedFmeasure(),'fm':Fmeasure(),'em':Emeasure()}
        self.normalize=normalize
        self.count=self.structural_count=0
        self.tp=self.fp=self.fn=self.tn=0
        self.mae=self.mean_probability=0.0
        self.region_detected={'SOD':0,'COD':0}
        self.region_total={'SOD':0,'COD':0}

    def step(self,probability,target,structural=True,semantic=None):
        probability=np.asarray(probability,dtype=np.float32)
        target=np.asarray(target,dtype=bool)
        if probability.shape!=target.shape or not np.isfinite(probability).all():
            raise ValueError('Invalid probability shape/values')
        if (probability<0).any() or (probability>1).any(): raise ValueError('Probability outside [0,1]')
        if self.normalize:
            probability=(probability*255).astype(np.uint8).astype(np.float32)/255
            low,high=float(probability.min()),float(probability.max())
            if high>low: probability=(probability-low)/(high-low)
        if structural:
            for metric in self.metrics.values(): metric.step(pred=probability,gt=target,normalize=False)
            self.structural_count+=1
        binary=probability>=0.5
        self.tp+=int(np.count_nonzero(binary&target));self.fp+=int(np.count_nonzero(binary&~target))
        self.fn+=int(np.count_nonzero(~binary&target));self.tn+=int(np.count_nonzero(~binary&~target))
        self.mae+=float(np.abs(probability-target).mean());self.mean_probability+=float(probability.mean());self.count+=1
        if semantic is not None:
            for key,class_id in (('SOD',1),('COD',2)):
                region=np.asarray(semantic)==class_id
                self.region_total[key]+=int(region.sum())
                self.region_detected[key]+=int(np.count_nonzero(region&binary))

    def compute(self):
        if not self.count: raise ValueError('No images accumulated')
        divide=lambda a,b:a/b if b else None
        result={'images':self.count,'MAE':self.mae/self.count,
                'binary_foreground_IoU':divide(self.tp,self.tp+self.fp+self.fn),
                'foreground_recall':divide(self.tp,self.tp+self.fn),
                'background_FPR':divide(self.fp,self.fp+self.tn),
                'foreground_missed_pixels':self.fn,'background_false_positive_pixels':self.fp,
                'mean_foreground_probability':self.mean_probability/self.count,
                'confusion_matrix':[[self.tn,self.fp],[self.fn,self.tp]],
                'region_recall':{k:{'GT_pixels':self.region_total[k],
                    'detected_pixels':self.region_detected[k],
                    'recall':divide(self.region_detected[k],self.region_total[k])} for k in ('SOD','COD')}}
        if self.structural_count:
            result.update(Salpha=float(self.metrics['sm'].get_results()['sm']),
                          Fbeta_w=float(self.metrics['wfm'].get_results()['wfm']),
                          maxFm=float(self.metrics['fm'].get_results()['fm']['curve'].max()),
                          maxEm=float(self.metrics['em'].get_results()['em']['curve'].max()))
        return result

BINARY_PROTOCOL={'threshold':0.5,'probability':'raw float32 sigmoid; no per-image normalization',
                 'metrics':'pysodmetrics 1.6.2; normalize=False',
                 'IoU':'sum pixel TP/FP/FN before division','MAE':'mean of per-image MAE',
                 'Scene-D':'MAE/background FPR; excluded from Scene-ABC structural aggregate',
                 'P0':'foreground before semantic correction; jointly trained in SegDis',
                 'Pu':'foreground after semantic correction; P0==Pu for independent FLR',
                 'region_recall':'Detected GT SOD/COD pixels / all pixels of that GT type; not type correctness'}
