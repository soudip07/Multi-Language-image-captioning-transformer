import pandas as pd
import numpy as np
from collections import Counter
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import pickle
import gc
import random
pd.set_option('display.max_colwidth', None)

max_seq_len = 33
vocab_size = 8360
k=3

"""**Define model architechture**"""

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        if self.pe.size(0) < x.size(0):
            self.pe = self.pe.repeat(x.size(0), 1, 1)
        self.pe = self.pe[:x.size(0), : , : ]

        x = x + self.pe
        return self.dropout(x)

class ImageCaptionModel(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size):
        super(ImageCaptionModel, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, 0.1)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model =  embedding_size, nhead = n_head)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer = self.TransformerDecoderLayer, num_layers = n_decoder_layer)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size , embedding_size)
        self.last_linear_layer = nn.Linear(embedding_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-initrange, initrange)

    def generate_Mask(self, size, decoder_inp):
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(decoder_input_mask == 1, float(0.0))

        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0

        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, encoded_image, decoder_inp):
        encoded_image = encoded_image.permute(1,0,2)


        decoder_inp_embed = self.embedding(decoder_inp)* math.sqrt(self.embedding_size)

        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)
        decoder_inp_embed = decoder_inp_embed.permute(1,0,2)


        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_Mask(decoder_inp.size(1), decoder_inp)
        decoder_input_mask = decoder_input_mask
        decoder_input_pad_mask = decoder_input_pad_mask
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool


        decoder_output = self.TransformerDecoder(tgt = decoder_inp_embed, memory = encoded_image, tgt_mask = decoder_input_mask, tgt_key_padding_mask = decoder_input_pad_mask_bool)

        final_output = self.last_linear_layer(decoder_output)

        return final_output,  decoder_input_pad_mask


def define_arch():
    



ictModel = ImageCaptionModel(16, 4, vocab_size, 512)
optimizer = torch.optim.Adam(ictModel.parameters(), lr = 0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience=2, verbose = True)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
min_val_loss = float('Inf')

model = torch.load('./BestModel', map_location=torch.device('cpu'))
transformer_model=model

"""**Load pretrained resnet18 model**"""

# # Load a pretrained ResNet-18
# resnet18 = torchvision.models.resnet18(pretrained=True)
# resnet18.eval()  # Set the model to evaluation mode

# # Remove the FC and AvgPool layers (we only want up to Layer 4)
# modules = list(resnet18.children())[:-2]  # Removes 'avgpool' and 'fc'
# feature_extractor = torch.nn.Sequential(*modules)

# #print(feature_extractor)

# Load ResNet18 model and extract features from the image
resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
resnet18.eval()

"""**Load vocab**"""

# Load the dictionaries from a file
def load_vocab(filepath):
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded from {filepath}")
    return vocab['index_to_word'], vocab['word_to_index']

filepath='./vocab.pkl'
index_to_word, word_to_index = load_vocab(filepath)
print(len(index_to_word), len(word_to_index))  # Verify the sizes

"""**Extract features from image using resnet18**"""

# def extract_image_features(image, feature_extractor):
#     """
#     Extracts feature maps from an image using a ResNet-18 (up to Layer 4).
#     Returns:
#     - torch.Tensor: Extracted feature map of shape (batch_size, 512, 7, 7) for a 224x224 image.
#     """
#     preprocess = torchvision.transforms.Compose([
#         torchvision.transforms.Resize((224, 224)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     if isinstance(image, torch.Tensor):
#         image_tensor = image.unsqueeze(0) if image.ndim == 3 else image
#     else:
#         image_tensor = preprocess(image).unsqueeze(0)

#     image_tensor = image_tensor.to(next(feature_extractor.parameters()))
#     with torch.no_grad():
#         features = feature_extractor(image_tensor)

#     return features

# Function to extract image features using the ResNet18 model (before final classification layer)
def extract_image_features(image, model):
    """ Extract features from the image using ResNet18 """
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model.conv1(image_tensor)  # First conv layer
        features = model.bn1(features)  # Batch norm
        features = model.relu(features)  # ReLU
        features = model.maxpool(features)  # Max pooling
        features = model.layer1(features)  # Layer 1
        features = model.layer2(features)  # Layer 2
        features = model.layer3(features)  # Layer 3
        features = model.layer4(features)  # Layer 4
        features = model.avgpool(features)  # Average Pooling
        features = features.view(features.size(0), -1)  # Flatten to a vector
    return features

"""**Generate caption**"""

# Generate caption for an image outside the dataset
def generate_caption(K, img_path, index_to_word, word_to_index):
    # Load and display the image
    image = Image.open(img_path).convert("RGB")
    plt.imshow(image)
    plt.axis("off")
    plt.show()


    img_embed = extract_image_features(image, resnet18)

    # img_embed is now a flattened vector of shape (batch_size, 512)
    img_embed = img_embed.unsqueeze(1)  # Add sequence length dimension: (batch_size, seq_len, num_features)

    # Prepare input sequence for caption generation
    input_seq = [word_to_index.get('<start>', 0)] * max_seq_len
    input_seq = torch.tensor(input_seq).unsqueeze(0)

    predicted_sentence = []
    model.eval()
    with torch.no_grad():
        for eval_iter in range(0, max_seq_len):
            # Forward pass through the trained captioning model
            output, padding_mask = model.forward(img_embed, input_seq)
            output = output[eval_iter, 0, :]

            # Perform top-K sampling
            values = torch.topk(output, K).values.tolist()
            indices = torch.topk(output, K).indices.tolist()
            next_word_index = random.choices(indices, values, k=1)[0]

            next_word = index_to_word.get(next_word_index, '<unk>')  # Use '<unk>' for unknown words

            # Update input sequence with the predicted word
            input_seq[:, eval_iter + 1] = next_word_index

            # Stop if the model predicts the end token
            if next_word == '<end>':
                break

            predicted_sentence.append(next_word)

    # Print the predicted caption
    print("\nPredicted Caption:")
    print(" ".join(predicted_sentence + ['.']))

def caption_this_image(img_path):
    cap = generate_caption(3, img_path, index_to_word, word_to_index)
    return cap
