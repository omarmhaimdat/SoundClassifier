import turicreate as tc
from os.path import basename

# Set the number of GPUs used (in this case all)
# -1 -> Every GPU available
# 1 -> One GPU
# 0 -> NO GPU only CPU
tc.config.set_num_gpus(-1)

# Load the audio data and meta data.
data = tc.load_audio('/content/ESC-50/ESC-50-master/audio/')
meta_data = tc.SFrame.read_csv('/content/ESC-50/ESC-50-master/meta/esc50.csv')

# Join the audio data and the meta data.
data['filename'] = data['path'].apply(lambda p: basename(p))
data = data.join(meta_data)

# Drop all records which are not part of the ESC-10.
data = data.filter_by('True', 'esc10')

# Make a train-test split, just use the first fold as our test set.
test_set = data.filter_by(1, 'fold')
train_set = data.filter_by(1, 'fold', exclude=True)

# Create the model.
model = tc.sound_classifier.create(train_set,
                                   target='category',
                                   feature='audio',
                                   max_iterations=100,
                                   custom_layer_sizes=[200, 200])

# Generate an SArray of predictions from the test set.
predictions = model.predict(test_set)

# Evaluate the model and print the results
metrics = model.evaluate(test_set)
print(metrics)

# Save the model for later use in Turi Create
model.save('SoundClassification.model')

# Export for use in Core ML
model.export_coreml('SoundClassification.mlmodel')