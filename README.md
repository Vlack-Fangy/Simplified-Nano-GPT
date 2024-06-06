# Simplified-Nano-GPT

This project aims to build a simplified version of nanoGPT, inspired by the work of Andrej Karpathy, to understand the transformer architecture and GPT (Generative Pre-trained Transformer) models practically. By following the principles outlined in the seminal paper "Attention is All You Need," this project constructs a basic architecture of a decoder-only GPT. The project utilizes the Shakespearian dataset to train the model, aspiring to create a competent document completer for generating Shakespearian scripts.

## Project Overview

### Objective
The main objective of this project is to demystify the inner workings of transformer architectures and GPT models by implementing a simplified version of nanoGPT. This hands-on approach aims to deepen the understanding of key concepts such as self-attention, positional encoding, and the training process involved in creating language models.

### Dataset
We use the Shakespearian dataset, which consists of texts from William Shakespeare's works. This dataset is chosen to provide a distinct and challenging corpus for the model, pushing it to learn the unique style and structure of Shakespearian English. The ultimate goal is to enable the model to generate coherent and stylistically accurate Shakespearian text.

### Architecture
Following the decoder-only GPT architecture, our implementation incorporates the following components:
- **Tokenization**: Converting raw text into tokens that the model can process.
- **Embedding**: Mapping tokens to high-dimensional vectors.
- **Positional Encoding**: Adding positional information to the token embeddings to retain the order of tokens.
- **Self-Attention Mechanism**: Allowing the model to focus on different parts of the input sequence to generate the output.
- **Feed-Forward Neural Network**: Transforming the attention output to the desired shape.
- **Layer Normalization**: Improving training stability and performance.
- **Output Layer**: Producing the probability distribution over the vocabulary for the next token prediction.

### Implementation
The project is implemented in Python using PyTorch, a powerful and flexible deep learning library. Key steps in the implementation include:
- **Data Preprocessing**: Preparing the Shakespearian dataset by tokenizing the text and creating input-output pairs for training.
- **Model Construction**: Building the transformer architecture with multiple layers of self-attention and feed-forward networks.
- **Training Loop**: Defining the loss function, optimizer, and training loop to iteratively improve the model's performance.
- **Evaluation and Inference**: Assessing the model's ability to generate Shakespearian text and making adjustments to enhance its performance.

### Challenges and Learnings
Throughout the project, several challenges were encountered and addressed:
- **Understanding Transformer Components**: Gaining a deep understanding of each component of the transformer architecture and how they interact.
- **Efficient Tokenization**: Developing an effective tokenization strategy that balances vocabulary size and model performance.
- **Training Stability**: Implementing techniques like layer normalization and learning rate scheduling to ensure stable and efficient training.
- **Text Generation**: Fine-tuning the model to generate coherent and contextually appropriate Shakespearian text.

### Results
By the end of the project, we successfully trained a simplified nanoGPT model capable of generating Shakespearian-style text. The model demonstrates a reasonable understanding of the structure and style of Shakespeare's works, producing outputs that are coherent and stylistically accurate. While there is room for further improvement, the project provides a solid foundation for understanding and building GPT models.

## Conclusion
This Simplified-Nano-GPT project serves as an educational tool for exploring the intricacies of transformer architectures and GPT models. By implementing a decoder-only GPT and training it on the Shakespearian dataset, we gain practical insights into the design and functioning of modern language models. This project not only demystifies the technology behind GPT but also lays the groundwork for future explorations and advancements in natural language processing.

---

We would like to extend our heartfelt thanks to Andrej Karpathy for his inspiration and guidance, which made this educational experience possible.
