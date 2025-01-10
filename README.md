Music Genre Classification:

This program contains machine learning models that classify an input 30-second music clip into the corresponding genre chosen from 10 popular music genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock. The models are trained on the GTZAN dataset which contains 100 30-second clips corresponding to each genre. The model is preprocessed using the librosa module, where the relevant features are extracted from the .wav files and written onto a JSON file for accessibility. In this project, we chose to use the Mel-Frequency Cepstral Coefficients(MFCC) as the primary feature for classification by the model. 

For feature engineering, we created 3 CNN models trained on MFCC, MFCC with delta-MFCC (first derivative of MFCC), and MFCC along with delta_MFCC and delta_2 MFCC (first and second derivative of MFCC) to evaluate the most effective model for music genre classification to be used in the next expensive step. By evaluating over 30 iterations, we concluded that the higher-order MFCC allows for greater data stability and improved classification accuracy. The results of the training sequence on the 3 CNN models are depicted below:

<img width="575" alt="Screenshot 2025-01-10 at 1 17 39 PM" src="https://github.com/user-attachments/assets/30ff83ab-e500-491c-8048-b5912103d457" />

After feature selection, we created and trained an RNN-LSTM model trained on MFCC, delta-MFCC, and delta_2-MFCC for the music genre classification model, with 444,298 parameters, which after running for 20 iterations with early-stoppage and dynamic learning rates resulted in a testing accuracy of 75.04%. Further epochs would've resulted in higher accuracy, but the training time also increases drastically for higher iterations. 

Required modules:

    numpy
    scikit-learn
    tensorflow
    librosa

To run the .ipynb files, install the above-mentioned modules after navigating to the file after cloning the repository, and run the .pynb files.
    
