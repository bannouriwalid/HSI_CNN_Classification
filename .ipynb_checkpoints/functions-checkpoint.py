import numpy as np
import os
import scipy.io
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
from tensorflow.keras import utils as np_utils
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Flatten, Input, Conv3D, Conv2D, Reshape, Dropout
from keras.models import Model, model_from_json
from keras.utils import to_categorical

# chargement des fichier groundtruth
def load_files(names):
    loaded_files = []
    for i, j in names.items():
        loaded_files.append(scipy.io.loadmat(i)[j])
    return loaded_files

# visualisation des images groudtruth
def visualize_groud_truth(files, classes, titles):
    plt.figure(figsize=(12, 6))
    n = len(files)
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(files[i], interpolation='nearest')  # colormap pour afficher différentes classes
        plt.colorbar(ticks= range(0,classes[i]))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def create_false_color(data, bands):
    false_color = np.dstack([data[:, :, b] for b in bands])
    false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min())
    return false_color

def plot_random_pixel_signature(data, labels, wavelengths):
    h, w = labels.shape
    while True:
        i, j = np.random.randint(0, h), np.random.randint(0, w)
        if labels[i, j] != 0:
            break
    spectrum = data[i, j, :]
    return spectrum, i, j

def plot_comprehensive_view(data, labels, wavelengths, bands, figsize):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
        
    # False color composite
    false_color = create_false_color(data, bands)
    axes[0, 0].imshow(false_color)
    axes[0, 0].set_title('False Color Composite')
        
    # random_pixel_signature
    signature, i, j = plot_random_pixel_signature(data, labels, wavelengths)
    axes[0, 1].plot(wavelengths, signature, color='red')
    axes[0, 1].set_title(f'Signature spectrale du pixel ({i}, {j}) (classe {labels[i, j]})')
    axes[0, 1].set_xlabel('Longueur d’onde (nm)')
    axes[0, 1].set_ylabel('Réflectance')
    axes[0, 1].grid(True)
        
    # Single band visualization
    mid_band = data.shape[2] // 2
    im = axes[1, 0].imshow(data[:, :, mid_band], cmap='gray')
    axes[1, 0].set_title(f'Single Band ({int(wavelengths[mid_band])}nm)')
        
    # Spectral variance
    spectral_variance = np.std(data, axis=2)
    im = axes[1, 1].imshow(spectral_variance)
    axes[1, 1].set_title('Spectral Variance')
        
    plt.tight_layout()
    plt.show()

def plot_class_distribution(labels, class_names):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(12, 4))
    plt.bar(unique, counts, color='skyblue')
    # Axis labels and title
    plt.xticks(unique, unique)
    plt.xlabel("Numéro de la classe")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Distribution des classes")
    legend_labels = [f"{i} - {class_names[i]}" for i in unique]
    legend_handles = [mpatches.Patch(color='white', label=label) for label in legend_labels]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()

def plot_sample_spectra(data, labels, wavelengths, class_names, classes):
    plt.figure(figsize=(12, 4))
    legend_handles = []
    for class_id in classes:
        mask = labels == class_id
        mean_spectrum = np.mean(data[mask], axis=0)
        # Plot and capture the line
        line, = plt.plot(wavelengths, mean_spectrum, label=class_names[class_id])
        # Use the same color for the legend
        legend_handles.append(Line2D([0], [0], color=line.get_color(), label=f"{class_id} - {class_names[class_id]}"))
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Réflectance moyenne")
    plt.title("Spectres moyens par classe")
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def visualize_dataset(data, labels, wavelengths, class_names, bands, figsize, classes):
    plot_comprehensive_view(data, labels, wavelengths, bands, figsize)
    plot_class_distribution(labels, class_names)
    plot_sample_spectra(data, labels, wavelengths, class_names, classes)

def plot_number_of_components(data):
    h, w, b = data.shape # reshape to (N_pixels, B)
    reshaped_data = data.reshape(-1, b)
    scaler = StandardScaler(with_std=False)  # Just mean-centering
    centered_data = scaler.fit_transform(reshaped_data)
    pca = PCA()
    pca.fit(centered_data)
    cum_variance = np.cumsum(pca.explained_variance_ratio_)
    tau = [0.9, 0.925, 0.95, 0.975, 0.99, 0.999]
    n_pca = []
    for t in tau:
        n_components = np.searchsorted(cum_variance, t) + 1
        n_pca.append(n_components)
    plt.figure(figsize=(10, 5))
    plt.scatter(tau, n_pca, color='blue')
    plt.plot(tau, n_pca, linestyle='--', color='gray')
    plt.xlabel("Seuil de variance cumulée")
    plt.ylabel("Nombre de composantes principales")
    plt.xticks(tau, [f"{t*100}%" for t in tau])  
    plt.yticks(n_pca)
    plt.grid(True)
    plt.title("Composantes PCA nécessaires selon le seuil de variance")
    plt.tight_layout()
    plt.show()

def apply_PCA(X, n_components):
    h, w, b = X.shape
    X_reshaped = X.reshape(-1, b)
    scaler = StandardScaler(with_std=False)  # Just mean-centering
    centered_data = scaler.fit_transform(X_reshaped)
    pca = PCA(n_components=n_components, whiten=True)
    X_pca = pca.fit_transform(centered_data)
    X_pca_reshaped = X_pca.reshape(h, w, n_components)
    return X_pca_reshaped, pca

def plot_first_n_components(data, n_components):
    plt.figure(figsize=(15, 6))
    if n_components <= data.shape[2]:
        for i in range(n_components):
            plt.subplot(1, 5, i+1)
            plt.imshow(data[:, :, i], cmap='gray')
            plt.title(f'PC{i+1}')
            plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01)
        plt.show()

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def preprocess_hyperspectral_data(X, y, num_components, test_ratio=0.2, windowSize=25):
    # Applying PCA
    X_reduced, pca = apply_PCA(X, num_components)
    # Creating the cube
    X, y = createImageCubes(X_reduced, y, windowSize)
    # Splitting train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=345, stratify=y)
    # Final shaping
    X_train = X_train.reshape(-1, windowSize, windowSize, num_components, 1)
    y_train = np_utils.to_categorical(y_train)
    X_test = X_test.reshape(-1, windowSize, windowSize, num_components, 1)
    y_test = np_utils.to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test

def build_3d_2d_cnn(S, L, output_units):
    """
    S : taille spatiale du patch (ex: 5)
    L : nombre de bandes spectrales
    output_units : nombre de classes en sortie
    """
    input_layer = Input((S, S, L, 1))

    # Convolutions 3D
    x = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu', padding='valid')(input_layer)
    x = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu', padding='valid')(x)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='valid')(x)
    shape = x.shape
    x = Reshape((x.shape[1], x.shape[2], x.shape[3] * x.shape[4]))(x)

    # Convolution 2D
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='valid')(x)

    # Dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(output_units, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def compile_train(model, X_train, y_train, X_test, y_test, optimizer='adam', batch_size=64, epochs=20):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
    return history

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model_performance(model, X_test, y_test):
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)  # One-hot → classes entières
    # Overall Accuracy (OA)
    oa = accuracy_score(y_true, y_pred_classes)
    # Matrice de confusion
    conf_mat = confusion_matrix(y_true, y_pred_classes)
    # Accuracy par classe (AA)
    per_class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)
    aa = per_class_accuracy.mean()
    # Coefficient de Kappa
    k = cohen_kappa_score(y_true, y_pred_classes)
    # Résumé global
    print(f"\nOverall Accuracy (OA) : {oa*100:.2f}%")
    print(f"Average Accuracy (AA) : {aa*100:.2f}%")
    print(f"Kappa coefficient (K) : {k:.4f}")
    # Affichage des résultats par classe
    classes = [f"Classe{i+1}" for i in range(len(per_class_accuracy))]
    Accuracy = (per_class_accuracy * 100).round(2)

    f_classes = f""
    f_accracies = f""
    for i in range(len(classes)):
        f_classes += f"{classes[i]:<9}"
        f_accracies += f"{Accuracy[i]:<9}"

    print("\n=== Tableau d’Accuracy par classe ===\n")
    print(f_classes)
    print(f_accracies)

    return conf_mat, classes

def plot_conf_matrix(conf_mat, classes):
    # Heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Matrice de confusion")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()

def Patch(data, height_index, width_index, patch_size):
    height_slice = slice(height_index, height_index+ patch_size)
    width_slice = slice(width_index, width_index+ patch_size)
    patch = data[height_slice, width_slice, :]
    return patch

def display_prediction_vs_groundtruth(model, X, n_components, y_gt, windowSize):
    X, pca = apply_PCA(X, n_components)
    X = padWithZeros(X, windowSize // 2)

    height, width = y_gt.shape
    prediction_image = np.zeros((height, width))
    prediction_hole_image = np.zeros((height, width))  # Initialize this

    all_patches = []
    patches = []
    positions = []
    
    for i in range(height):
        for j in range(width):
            image_patch = Patch(X, i, j, windowSize)
            all_patches.append(image_patch)
            if y_gt[i, j] != 0:
                patches.append(image_patch)
                positions.append((i, j))

    # Predict on labeled points
    X_test_image = np.array(patches).astype('float32')[..., np.newaxis]
    predictions = model.predict(X_test_image, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1) + 1

    for idx, (i, j) in enumerate(positions):
        prediction_image[i, j] = predicted_classes[idx]

    # Predict on full image (holes included)
    all_patches_array = np.array(all_patches).astype('float32')[..., np.newaxis]
    full_predictions = model.predict(all_patches_array, verbose=1)
    full_predicted_classes = np.argmax(full_predictions, axis=1) + 1

    # Fill prediction_hole_image
    idx = 0
    for i in range(height):
        for j in range(width):
            prediction_hole_image[i, j] = full_predicted_classes[idx]
            idx += 1

    # Compute error image
    error_image = np.zeros((height, width))
    mask = y_gt != 0
    error_image[mask] = (prediction_image[mask] != y_gt[mask]).astype(int)
    
    # Affichage des résultats
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(prediction_image, interpolation='none')
    plt.title("Prédiction sur les classes")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(y_gt, interpolation='none')
    plt.title("Vérité terrain")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(error_image, interpolation='none')
    plt.title("Erreur de classification")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(prediction_hole_image, interpolation='none')
    plt.title("Prédiction compléte")
    plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01)
    plt.show()

def save_keras_model(model, filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename + ".weights.h5")

def load_keras_model(filename):
    # load json and create model
    json_file = open(filename + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename + ".weights.h5")
    return loaded_model