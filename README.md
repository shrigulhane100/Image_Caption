# AI Image Captioning Tool

The Image Caption project uses the **ViT-GPT2** model to generate captions for images. It uses the **transformers** library and **gradio** for the interface. The file contains the following functions and variables:
- `predict(image,max_length=64, num_beams=4)`: This function takes an image as input and returns a caption for the image. It uses the ViT-GPT2 model to generate the caption.
- `device='cpu'`: This variable sets the device to CPU.
- `encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"`: This variable sets the encoder checkpoint path.
- `decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"`: This variable sets the decoder checkpoint path.
- `model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"`: This variable sets the model checkpoint path.
- `feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)`: This variable extracts features from images using the ViT model.
- `tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)`: This variable tokenizes captions using the GPT2 tokenizer.
- `model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)`: This variable loads the ViT-GPT2 model.

 
 ## Results from 3 sample images: 
- a man in a soccer uniform kicking a soccer ball 
- two horses are standing in a field
- a collage of photos showing a men and women with a sign 

Check out the configuration reference at https://huggingface.co/docs/hub/spaces#reference
