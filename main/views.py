from django.shortcuts import render, HttpResponseRedirect
from .forms import ImageUpload
import os
import tensorflow as tf
import json
from tensorflow.keras.preprocessing import image
import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from PIL import Image


model = tf.keras.models.load_model(os.path.join(
    settings.BASE_DIR, 'cat_breed_classifier.keras'))


with open(os.path.join(settings.BASE_DIR, 'class_indices.json'), 'r') as f:
    class_indices = json.load(f)


class_labels = {v: k for k, v in class_indices.items()}


def home(request):
    if request.method == 'POST':
        form = ImageUpload(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/')
    else:
        form = ImageUpload()

    return render(request, 'home.html', {'form': form})


def prepare_image(img_path):
    # Получаем расширение файла
    supported_formats = ['jpeg', 'jpg', 'png', 'webp']
    img_format = img_path.split('.')[-1].lower()

    if img_format not in supported_formats:
        raise ValueError(
            "Image format isn't appropriate. You should load .jpeg, .jpg, .png, .webp")

    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict_breed(img_path, model, threshold=0.5):
    try:
        img_array = prepare_image(img_path)

        predictions = model.predict(img_array)

        predicted_class_index = np.argmax(predictions)
        predicted_class_prob = predictions[0][predicted_class_index]

        print(f"Predicted class index: {predicted_class_index}")
        print(f"Predicted class probability: {predicted_class_prob}")

        if predicted_class_prob < threshold:
            predicted_class = class_labels.get(
                predicted_class_index, "Unknown class")
            return f"NO ANSWER. But AI gives a {predicted_class_prob * 100:.2f}% probability it's {predicted_class}"
        else:
            predicted_class = class_labels.get(
                predicted_class_index, "Unknown class")
            return f"Predicted breed: {predicted_class}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"


def clear_image_folder():
    
    for filename in os.listdir(settings.MEDIA_ROOT):
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def classify_cat(request):
    clear_image_folder()

    breed = None
    image_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        img_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
        try:
            with default_storage.open(img_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            breed = predict_breed(img_path, model)
            image_url = default_storage.url(img_path)
        except Exception as e:
            breed = f"An error occurred while processing the image: {str(e)}"

    return render(request, 'home.html', {
        'breed': breed,
        'image_url': image_url,
        'form': ImageUpload()
    })
