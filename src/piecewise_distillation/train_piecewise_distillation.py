import tensorflow as tf
from dataloader import DataLoader
import matplotlib.pyplot as plt
from unet import UNet

input_dir = '../distillation/dataset/Sony/short/'
gt_dir = '../distillation/dataset/Sony/long/'

def piecewise_distill(teacher_model, student_model, piecewise_train_steps, full_train_steps, batch_size=8, print_every=10, graph=True):
    """
    Perform network distillation on the given models
    :param teacher_model: The model to sample the student targets from
    :param student_model: The model to train on teajcher targets
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
    # The outputs we will use for training
    piecewise_outputs = [teacher_model.piece_0_out, teacher_model.piece_1_out, teacher_model.output]


    # Iterate over pieces
    for i in range(3):
        print("Training piece " + str(i + 1) + " of U-Net")
        # Iterate over training steps
        for train_step in range(piecewise_train_steps):
            # Sample a batch
            input_batch, _ = dataloader.get_next_batch()

            # Get the target values from the teacher model
            feed_dict = {
                teacher_model.input: input_batch
            }

            targets = teacher_model.sess.run(piecewise_outputs[i], feed_dict=feed_dict)

            # TODO: Add more complex learning rate
            # Make a training step on these targets
            loss_value = student_model.train_step(input_batch, targets, student_model.sess, piece=i)

            if train_step % print_every == 0:
                # Print the training loss every <print_every> steps
                print("Loss value on step " + str(train_step) + ": " + str(loss_value) + " training piece " + str(i + 1))

    # Train the full model together
    # Iterate over train steps
    print("\n\nTraining full model...")
    for train_step in range(full_train_steps):
        # Sample a batch
        input_batch, _ = dataloader.get_next_batch()

        # Get the target values from the teacher model
        targets = teacher_model.predict(input_batch, teacher_model.sess)

        # TODO: Add more complex learning rate
        # Make a training step on these targets
        loss_value = student_model.train_step(input_batch, targets, student_model.sess)

        if train_step % print_every == 0:
            # Print the training loss every <print_every> steps
            print("Loss value on step " + str(train_step) + ": " + str(loss_value))

        losses += [loss_value]

    # Save the student model
    student_model.save_model(save_name="Distilled" + str(student_model.start_channel_depth))

    # Plot the loss if we want to
    if graph:
        plt.plot(list(range(len(losses))), losses)
        plt.show()

def main():
    # Build the student model
    # Change these as you see fit
    STUDENT_MODEL_SIZE = 4
    PIECEWISE_TRAIN_STEPS = 2000
    FULL_TRAIN_STEPS = 500

    student_model = UNet(start_channel_depth=STUDENT_MODEL_SIZE, student=True)
    teacher_model = UNet(start_channel_depth=32)

    teacher_model.load_model(32)

    piecewise_distill(teacher_model, student_model, PIECEWISE_TRAIN_STEPS, FULL_TRAIN_STEPS)


if __name__ == '__main__':
    main()