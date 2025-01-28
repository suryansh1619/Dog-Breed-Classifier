import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image

model = tf.keras.models.load_model('dog_breed_classifier.keras')

def preprocess_image(file):
    img = Image.open(BytesIO(file.read()))
    img = img.resize((300, 300)) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return img_array

def predict_breed(file):
    img_array = preprocess_image(file)
    predictions = model.predict(img_array)[0]
    breeds = decode_prediction(predictions)
    return breeds

def decode_prediction(predictions):
    breed_names = [
        'Afghan Hound', 'African Hunting Dog', 'Airedale', 'American Staffordshire Terrier', 'Appenzeller',
        'Australian Terrier', 'Bedlington Terrier', 'Bernese Mountain Dog', 'Blenheim Spaniel', 'Border Collie',
        'Border Terrier', 'Boston Bull', 'Bouvier Des Flandres', 'Brabancon Griffon', 'Brittany Spaniel',
        'Cardigan', 'Chesapeake Bay Retriever', 'Chihuahua', 'Dandie Dinmont', 'Doberman', 'English Foxhound',
        'English Setter', 'English Springer', 'Entlebucher', 'Eskimo Dog', 'French Bulldog', 'German Shepherd',
        'German Short Haired Pointer', 'Gordon Setter', 'Great Dane', 'Great Pyrenees', 'Greater Swiss Mountain Dog',
        'Ibizan Hound', 'Irish Setter', 'Irish Terrier', 'Irish Water Spaniel', 'Irish Wolfhound', 'Italian Greyhound',
        'Japanese Spaniel', 'Kerry Blue Terrier', 'Labrador Retriever', 'Lakeland Terrier', 'Leonberg', 'Lhasa',
        'Maltese Dog', 'Mexican Hairless', 'Newfoundland', 'Norfolk Terrier', 'Norwegian Elkhound', 'Norwich Terrier',
        'Old English Sheepdog', 'Pekinese', 'Pembroke', 'Pomeranian', 'Rhodesian Ridgeback', 'Rottweiler', 'Saint Bernard',
        'Saluki', 'Samoyed', 'Scotch Terrier', 'Scottish Deerhound', 'Sealyham Terrier', 'Shetland Sheepdog', 'Shih Tzu',
        'Siberian Husky', 'Staffordshire Bullterrier', 'Sussex Spaniel', 'Tibetan Mastiff', 'Tibetan Terrier', 'Walker Hound',
        'Weimaraner', 'Welsh Springer Spaniel', 'West Highland White Terrier', 'Yorkshire Terrier', 'Affenpinscher', 'Basenji',
        'Basset', 'Beagle', 'Black And Tan Coonhound', 'Bloodhound', 'Bluetick', 'Borzoi', 'Boxer', 'Briard', 'Bull Mastiff',
        'Cairn', 'Chow', 'Clumber', 'Cocker Spaniel', 'Collie', 'Curly Coated Retriever', 'Dhole', 'Dingo', 'Flat Coated Retriever',
        'Giant Schnauzer', 'Golden Retriever', 'Groenendael', 'Keeshond', 'Kelpie', 'Komondor', 'Kuvasz', 'Malamute', 'Malinois',
        'Miniature Pinscher', 'Miniature Poodle', 'Miniature Schnauzer', 'Otterhound', 'Papillon', 'Pug', 'Redbone', 'Schipperke',
        'Silky Terrier', 'Soft Coated Wheaten Terrier', 'Standard Poodle', 'Standard Schnauzer', 'Toy Poodle', 'Toy Terrier',
        'Vizsla', 'Whippet', 'Wire Haired Fox Terrier'
    ]

    top_indices = predictions.argsort()[-5:][::-1]
    top_predictions = [{"breed": breed_names[i], "probability": float(predictions[i])} for i in top_indices]
    return top_predictions

