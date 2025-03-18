"""
Train an AutoEncoderKL with a controlnet in the decoder section
0. Find a copy of the AutoEncoderKL training code in the diffuser library to determine the basic training pipeline.
1. Load AutoEncoderKL from diffuser
2. Copy the decoder weights to generate decoder_controlnet, with pixel_value as input
3. Set the encoder to not require gradients, and the decoder and decoder_controlnet to require gradients
4. Start training.
5. Define the training loop and loss function
6. Implement logging to monitor training progress
7. Implement saving the model to disk"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from diffusers import AutoencoderKL

# 1. Load an AutoEncoderKL model from diffusers.
autoencoder = AutoencoderKL.from_pretrained("CompVis/autoencoder-kl")  # update with your desired model path
encoder = autoencoder.encoder
decoder = autoencoder.decoder

# 2. Copy the decoder weights to generate decoder_controlnet.
#    This copies the decoder architecture and weights so we can later add modifications if needed.
decoder_controlnet = copy.deepcopy(decoder)
# (Optionally modify decoder_controlnet to accept pixel_value as input if required by adding layers)

# 3. Set gradient requirements:
#    Freeze the encoder and allow gradients for the decoder and our controlnet.
for param in encoder.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False
for param in decoder_controlnet.parameters():
    param.requires_grad = True

# 5. Define the loss function and optimizer.
# Here we use a simple mean squared error loss.
loss_fn = nn.MSELoss()

# Combine the parameters of both decoders for optimization.
optimizer = optim.Adam(list(decoder.parameters()) + list(decoder_controlnet.parameters()), lr=1e-4)

# Dummy function to simulate a data loader (assumes each batch contains a dictionary with "pixel_values")
def get_train_loader():
    # Replace this with your actual dataloader
    # Here we simulate a dummy tensor of shape [batch, channels, height, width]
    dummy_images = torch.randn(8, 3, 256, 256)
    yield {"pixel_values": dummy_images}

# 4 & 6. Define the training loop with logging to monitor progress.
num_epochs = 5  # or any desired number of epochs

for epoch in range(num_epochs):
    for batch in get_train_loader():
        images = batch["pixel_values"]  # Assuming input images are normalized pixel values

        # Encode the images to latents.
        latents = encoder(images)[0]  # use output appropriately (the encoder might return a tuple)
        
        # Decode using both branches.
        reconstructed = decoder(latents)
        control_reconstructed = decoder_controlnet(latents)
        
        # Define a combined loss: here we're simply summing the reconstruction losses.
        loss_recon = loss_fn(reconstructed, images)
        loss_control = loss_fn(control_reconstructed, images)
        loss = loss_recon + loss_control
        
        # Backpropagation and optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging training progress.
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 7. Saving the trained models.
# Save the autoencoder (encoder + original decoder) and the new decoder_controlnet.
torch.save(autoencoder.state_dict(), "autoencoderKL.pth")
torch.save(decoder_controlnet.state_dict(), "decoder_controlnet.pth")