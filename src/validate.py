from distillation.dataloader import DataLoader
from channel_benchmarking.unet import UNet
import numpy as np
import glob

validation_input_dir = ""
validation_ground_truth_dir = ""
checkpoints_dir = ""

def validate_model(model):
    """
    Runs a forward pass on the validation set and returns loss
    :param model: The model to validate
    :return: The validation loss
    """

    # Build the dataloader for the current model
    dataloader = DataLoader(validation_input_dir, validation_ground_truth_dir, 1, is_validation=True)
    losses = []

    failed = False

    while not failed:
        try:
            input_batch, ground_truth_batch = dataloader.get_next_batch()
        except Exception:
            failed = True

        loss_value = model.evaluate(input_batch, ground_truth_batch, model.sess)
        losses += [loss_value]

    return np.mean(losses)

def get_file_metadata(filename):
    """
    Gets the UNet size and number of epochs from the filename
    :param filename: The filename to pass in
    :return: A tuple with (UNet size, Epochs trained for)
    """
    # Remove relative path and keep filename
    filename = filename.split("/")[-1]

    if filename[4] in ['2', '4', '8']:
        # This first character is the only one in the UNet size
        size = int(filename[4])
        epochs = int(filename.split(".")[0][5:])
    else:
        # The first two numbers are the UNet size
        size = int(filename[4:6])
        epochs = int(filename.split(".")[0][6:])

    return size, epochs


def main():
    """
    Runs through all saved models and validates them
    :return: None
    """
    save_info = []

    filenames = glob.glob(checkpoints_dir + "/*.meta")
    for filename in filenames:
        # Get the metadata for saving
        model_size, epochs = get_file_metadata(filename)
        # Build a model
        model = UNet()
        model.load_model(model_size, filename)

        validation_loss = validate_model(model)

        save_info += [[model_size, epochs, validation_loss]]


if __name__ == '__main__':
    main()