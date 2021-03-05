import ember
import os
path = os.path.abspath(os.path.join(os.path.expanduser('~'), ".julia/datadeps/EMBER/ember2018"))
print("Computing EMBER features")
ember.create_vectorized_features(path)
ember.create_metadata(path)
print("Done.")
