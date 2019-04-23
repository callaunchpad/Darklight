import tensorflow as tf
from dataloader import DataLoader
import matplotlib.pyplot as plt
from unet import UNet

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'

def distill_models(teacher_model, student_model, train_steps, batch_size=8, print_every=10, graph=True):
    """
    Perform network distillation on the given models
    :param teacher_model: The model to sample the student targets from
    :param student_model: The model to train on teacher targets
    :param train_steps: The number of training steps to train for
    :param batch_size: The size of batches to use during training
    :param print_every: How often to print the loss
    :param graph: Whether or not to graph the loss at the end of distillation
    :return: None
    """

    # Build the dataloader
    dataloader = DataLoader(input_dir, gt_dir, batch_size)

    # Keeps track of losses for plotting
    losses = []

    # Iterate over epochs
    for train_step in range(train_steps):
        # Sample a batch
        input_batch, _ = dataloader.get_next_batch()

        # Get the target values from the teacher model
        targets = teacher_model.predict(input_batch, teacher_model.sess)

        # TODO: Add more complex learning rate
        # Make a training step on these targets
        loss_value = student_model.train_step(input_batch, targets, student_model.sess)

        if train_step % print_every == 0:
            # Print the training loss every <print_every> steps
            print(f"Loss value on step: {train_step}: {loss_value}")

        losses += [loss_value]

    # Save the student model
    student_model.save_model(save_name="Distilled" + str(student_model.start_channel_depth))

    # Plot the loss if we want to
    if graph:
        plt.plot(list(range(len(losses))), losses)
        plt.show()
