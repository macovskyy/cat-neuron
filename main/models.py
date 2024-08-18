from django.db import models
from django.core.files.storage import FileSystemStorage

fs = FileSystemStorage(location='media/')

class Image(models.Model):
    image = models.ImageField(upload_to='images/')

