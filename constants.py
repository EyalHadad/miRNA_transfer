

EXTERNAL_PATH = "data/external"
RAW_PATH = "data/raw"
EXTERNAL_TRAIN_PATH = "data/external/train"
EXTERNAL_TEST_PATH = "data/external/test"
INTERIM_PATH = "data/interim"
PROCESSED_PATH = "data/processed"
PROCESSED_TRAIN_PATH = "data/processed/train"
PROCESSED_TEST_PATH = "data/processed/test"
MODELS_PATH = "models"
MODELS_OBJECTS_PATH = "models/objects"
MODELS_INTRA_TABELS = "models/intra_tabels"
MODELS_CROSS_ORG_TABELS = "models/cross_org_tabels"
MODELS_OBJECTS_TRANSFER_TABLES = "models/transfer_tables"
MODELS_OBJECTS_GRAPHS = "models/graphs"
MODELS_OUTPUT_PATH = "models/learning_output"
MODELS_FEATURE_IMPORTANCE = "reports/feature_importance"
MODELS_PREDICTION_PATH = "models/prediction"
MODELS_STD_PATH = "models/std"

TRANSFER_SIZE_LIST = [0, 100, 200, 500, 700]
# TRANSFER_SIZE_LIST = [0, 100, 200, 500, 700,1500,3000,5000,10000]
DATASETS = ['cow1', 'worm1', 'worm2', 'human1', 'human2', 'human3', 'mouse1', 'mouse2']
DATASETS2 = ['worm1', 'worm2', 'human1', 'mouse1']
SPECIES = ['worm', 'cow', 'human', 'mouse']
IMPORTANT_FEATURES = ['miRNAPairingCount_Seed_GU', 'miRNAMatchPosition_1','miRNAPairingCount_Total_GU',
'Energy_MEF_local_target', 'MRNA_Target_G_comp', 'MRNA_Target_GG_comp', 'miRNAMatchPosition_4',
'miRNAMatchPosition_5','miRNAPairingCount_Seed_bulge_nt','miRNAPairingCount_Seed_GC',
'miRNAMatchPosition_2','miRNAPairingCount_Seed_mismatch','miRNAPairingCount_X3p_GC',
                      'Seed_match_compact_interactions_all']
SEQUANCE_FEATURES = ['mRNA_start', 'label', 'mRNA_name',
                      'target sequence', 'microRNA_name', 'miRNA sequence', 'full_mrna']

FEATURES_TO_DROP = ['mRNA_start', 'label','mRNA_name','target sequence','microRNA_name','miRNA sequence','full_mrna',
'canonic_seed','duplex_RNAplex_equals','non_canonic_seed','site_start','num_of_pairs','mRNA_end','constraint']
# ,'Accessibility (nt=21, len=10)',
MODEL_INPUT_SHAPE = 490
TRAIN_EPOCHS = 20
# TRANS_EPOCHS = 10
XGBS_PARAMS = {
            "objective": ["binary:hinge"],
            "booster" : ["gbtree"],
            "eta" : [0.1],
            'gamma': [0.5],
            'max_depth': range(2, 4, 2),
            'min_child_weight': [1],
            'subsample': [0.6],
            'colsample_bytree': [0.6],
            "lambda" : [1],
            "n_jobs": [-1],

        }