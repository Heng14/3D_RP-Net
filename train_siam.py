import os
import glob

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import siam3dunet_model
from unet3d.model import testnet_model
from unet3d.training import load_old_model, train_model
from skimage.io import imsave, imread


config = dict()
config["image_shape"] = (64, 64, 16)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = (1,)  # the label numbers on the input image
config["n_base_filters"] = 16
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["dwi", "t1", "t1c", "t2"]
config["training_modalities"] = ["t1"] #config["all_modalities"]  # change this if you want to only use some of the modalities
#config["nb_channels"] = len(config["training_modalities"])
config["nb_channels"] = 1

mode = config["training_modalities"][0]

if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 239
config["validation_batch_size"] = 60
config["n_epochs"] = 1000  # cutoff the training after this many epochs
config["patience"] = 100  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 1000  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 1e-2 #5e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = True  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = 0.5  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file0"] = os.path.abspath(f"siam_data0_{mode}.h5")
config["data_file1"] = os.path.abspath(f"siam_data1_{mode}.h5")
config["model_file"] = os.path.abspath(f"siam_model_{mode}.h5")
config["training_file"] = os.path.abspath(f"siam_training_ids_t1.pkl")
config["validation_file"] = os.path.abspath(f"siam_validation_ids_t1.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.


def fetch_training_data_files(return_subject_ids=False):
    training_data_files0 = list()
    subject_ids0 = list()
    training_data_files1 = list()
    subject_ids1 = list()
    train0 = glob.glob(os.path.join(os.path.dirname(__file__), "twostage_data_300", "preprocessed", "pre", "*"))
    train1 = glob.glob(os.path.join(os.path.dirname(__file__), "twostage_data_300", "preprocessed", "post", "*"))
    #train0 = glob.glob(os.path.join(os.path.dirname(__file__), "twostage_data_300", "preprocessed", "pre", "*"))
    #train1 = glob.glob(os.path.join(os.path.dirname(__file__), "twostage_data_300", "preprocessed", "post", "*"))

    train0.sort(key = lambda x: x.split('/')[-1].split('-')[-1])
    train1.sort(key = lambda x: x.split('/')[-1].split('-')[-1])

    train0_index = [i.split('/')[-1].split('-')[-1] for i in train0]
    train1_index = [i.split('/')[-1].split('-')[-1] for i in train1]

    assert train0_index == train1_index, print ('train0 and train1 sort not equal !!!')

    for subject_dir in train0:
        subject_ids0.append(os.path.basename(subject_dir))

        subject_files = list()
        for modality in config["training_modalities"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        subject_files.append(os.path.join(subject_dir, "truth.nii"))
        training_data_files0.append(tuple(subject_files))

    for subject_dir in train1:
        subject_ids1.append(os.path.basename(subject_dir))

        subject_files = list()
        for modality in config["training_modalities"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        subject_files.append(os.path.join(subject_dir, "truth.nii"))
        training_data_files1.append(tuple(subject_files))

    assert len(subject_ids0) == len(subject_ids1), print ('len subject_ids0 and subject_ids1 are not equal !!!')

    training_data_files = [training_data_files0, training_data_files1]
    subject_ids = [subject_ids0, subject_ids1]

    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files


def main(overwrite=False):

    # convert input images into an hdf5 file
    if overwrite or not (os.path.exists(config["data_file0"]) and os.path.exists(config["data_file1"])):

        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)
        training_files0, training_files1 = training_files
        subject_ids0, subject_ids1 = subject_ids

        if not os.path.exists(config["data_file0"]):
            write_data_to_file(training_files0, config["data_file0"], image_shape=config["image_shape"], subject_ids=subject_ids0)
        if not os.path.exists(config["data_file1"]):
            write_data_to_file(training_files1, config["data_file1"], image_shape=config["image_shape"], subject_ids=subject_ids1)

    data_file_opened0 = open_data_file(config["data_file0"])
    data_file_opened1 = open_data_file(config["data_file1"])


    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = siam3dunet_model(input_shape=config["input_shape"], n_labels=config["n_labels"], initial_learning_rate=config["initial_learning_rate"], n_base_filters=config["n_base_filters"])

        #model = testnet_model(input_shape=config["input_shape"], n_labels=config["n_labels"], initial_learning_rate=config["initial_learning_rate"], n_base_filters=config["n_base_filters"])


        #if os.path.exists(config["model_file"]):
        #    model = load_weights(config["model_file"])

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened0,
        data_file_opened1,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])


    '''
    train_data = []
    train_label = []
    for i in range(n_train_steps):
        a, b = next(train_generator)
        train_data.append(a)
        train_label.append(b)

        a0, a1 = a

        for i in range(len(a0[0,0,0,0,:])):
            a0_0 = a0[0,2,:,:,i]
            if a0_0.min() == a0_0.max():
                a0_0 = a0_0 - a0_0
            else:                
                a0_0 = (a0_0-a0_0.min())/(a0_0.max()-a0_0.min())
        #print (a0_0.shape)
        #print (a0_0.max())
        #print (a0_0.min())
            imsave(f'vis_img/{i}.jpg', a0_0)
        raise
    '''    

    test_data, test_label = next(validation_generator)
    test_g = (test_data, test_label)

    train_data, train_label = next(train_generator)
    train_g = (train_data, train_label)


    if not overwrite and os.path.exists(config["model_file"]):

        txt_file = open(f"output_log.txt","w")

        #res = model.evaluate(test_data, test_label)
        #print (res)
        pre = model.predict(test_data)
        #print ([i for i in pre[0]])
        #print ([int(i) for i in test_label[0]])
        for i in range(len(pre[0])):
            txt_file.write(str(pre[0][i][0])+' '+str(test_label[0][i])+"\n")

        pre_train = model.predict(train_data)
        for i in range(len(pre_train[0])):
            txt_file.write(str(pre_train[0][i][0])+' '+str(train_label[0][i])+"\n")

        txt_file.close()
        raise

    # run training

    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=test_g,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])

    '''
    for i in range(len(train_label)):
        #scores = model.evaluate(train_data[i], train_label[i], verbose=1)
        scores = model.predict(train_data[i])
        print (len(scores[0]))
    '''

    data_file_opened0.close()
    data_file_opened1.close()

if __name__ == "__main__":
    main(overwrite=config["overwrite"])
