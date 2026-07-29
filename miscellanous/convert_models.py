
# If you want to use this, use it after putting this in the /root 

import sys
import os
import pickle
import joblib
from pathlib import Path


import sklearn.linear_model

sys.modules['sklearn.linear_model.logistic'] = sklearn.linear_model
sys.modules['sklearn.linear_model.stochastic_gradient'] = sklearn.linear_model



class DummyClass:
    pass


sys.modules['rationale_CNN'] = type(sys)('rationale_CNN')
sys.modules['rationale_CNN'].RationaleCNN = DummyClass
sys.modules['rationale_CNN'].RationaleCNN_Baseline = DummyClass

sys.modules['sample_size_NN'] = type(sys)('sample_size_NN')
sys.modules['sample_size_NN'].SampleSizeNeuralNet = DummyClass

sys.modules['vectorizer'] = type(sys)('vectorizer')
sys.modules['vectorizer'].FeatureVectorizer = DummyClass


try:
    import networkx
except ImportError:
    print("⚠️ 'networkx' not found. Some MeSH files might fail, but are likely unused.")
    sys.modules['networkx'] = type(sys)('networkx')



def convert_models():
    data_dir = Path("data")
    
    
    extensions = ["*.pickle", "*.pck", "*.p"]
    files_to_convert = []
    for ext in extensions:
        files_to_convert.extend(data_dir.rglob(ext))

    print(f"Found {len(files_to_convert)} files to convert.\n")

    for pkl_path in files_to_convert:
        try:
          
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
          
            joblib_path = pkl_path.with_suffix(".joblib")
            joblib.dump(data, joblib_path, compress=3)
            
            old_size = os.path.getsize(pkl_path) / (1024 * 1024)
            new_size = os.path.getsize(joblib_path) / (1024 * 1024)
            reduction = 100 - (new_size / old_size * 100) if old_size > 0 else 0
            
            print(f"✅ {pkl_path.name:<30} | {old_size:.1f}MB -> {new_size:.1f}MB ({reduction:.0f}% smaller)")
            
        except Exception as e:
        
            if "svm_cnn" in pkl_path.name:
                 print(f"❌ CRITICAL FAILURE: {pkl_path.name} | Error: {e}")
                 print("   ^ This file is needed for RCT predictions! Check the patch.")
            else:
                 print(f"⚠️ Skipped: {pkl_path.name:<25} | Reason: {e}")

    print("\n--- Conversion Complete ---")

if __name__ == "__main__":
    convert_models()