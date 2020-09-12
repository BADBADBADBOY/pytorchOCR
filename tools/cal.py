import sys
sys.path.append('./')
from tools.cal_rescall.script import cal_recall_precison_f1
from tools.cal_rescall.cal_det import cal_det_metrics


result = cal_recall_precison_f1('/src/notebooks/detect_text/icdar2015/ch4_test_gts/','/src/notebooks/detect_text/PytorchOCR3/result/result_txt')
print(result)

out = cal_det_metrics('/src/notebooks/detect_text/icdar2015/ch4_test_gts/', '/src/notebooks/detect_text/PytorchOCR3/result/result_txt')
print(out)