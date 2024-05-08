# Celebrity Face Matcher

## Description
Celebrity Face Matcher is a fun project that uses Convolutional Neural Networks (CNNs) to match your face with that of a celebrity. This project utilizes deep learning techniques to analyze facial features and find the closest match among a database of celebrities.

## Features
- Upload your photo and find out which celebrity you resemble the most.
- Built-in database of celebrities for comparison.
- Easy-to-use web interface.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Flask
- [Bollywood Celeb Localized Face Dataset Extended](https://www.kaggle.com/datasets/sroy93/bollywood-celeb-localized-face-dataset-extended)

## Installation
1. Clone this repository to your local machine.
2. Download the [Bollywood Celeb Localized Face Dataset Extended](https://www.kaggle.com/datasets/sroy93/bollywood-celeb-localized-face-dataset-extended) and extract it into the `dataset` folder.

## Usage
1. Run the Flask web server:
    ```
    python app.py
    ```
2. Navigate to `http://localhost:5000` in your web browser.
3. Upload your photo and wait for the result!

### Note:
- Celebrity Face Matcher uses cosine similarity to find the closest match between your face and the celebrities in the dataset.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- This project uses the VGGFace model for celebrity face recognition.
- Thanks to the developers of TensorFlow, Keras, and Flask for providing excellent libraries.

## Disclaimer
This project is for educational and entertainment purposes only. The resemblance detected by the model may not be accurate and should not be taken seriously.

## Author
[Rugved] - [Your GitHub Profile](https://github.com/rugvedp)
