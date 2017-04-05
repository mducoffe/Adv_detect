#python test_adv_cnn.py --option 'szegedy' --fisher 0 >> run_szegedy_classif_fisher0
#python test_adv_cnn.py --option 'szegedy' --fisher 1 >> run_szegedy_classif_fisher1
#python test_adv_cnn.py --option 'goodfellow' --fisher 0 >> run_goodfellow_classif_fisher0
#python test_adv_cnn.py --option 'goodfellow' --fisher 1 >> run_goodfellow_classif_fisher1
python test_adv_cnn.py --option 'deepfool' --fisher 0 >> run_deepfool_classif_fisher0
#python test_adv_cnn.py --option 'deepfool' --fisher 1 >> run_deepfool_classif_fisher1
