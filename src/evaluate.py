from dvclive import Live
import numpy as np
from numpy.lib.type_check import imag
import tensorflow, os, json, pickle, yaml
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageFont, ImageDraw

live = Live()
OUTPUT_DIR = "output"
fpath = os.path.join(OUTPUT_DIR, "data.pkl")
with open(fpath, "rb") as fd:
    data = pickle.load(fd)
(x_train, y_train),(x_test, y_test) = data
labels = y_test.astype(int)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
input_size = image_size * image_size

# Model specific code
raw_x_test = x_test.reshape(-1, 28, 28, 1)
x_test = raw_x_test.astype('float32') / 255
# End of Model specific code

model_file = os.path.join(OUTPUT_DIR, "model.h5")
model = tensorflow.keras.models.load_model(model_file)

metrics_dict = model.evaluate(x_test, y_test, return_dict=True)

METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.json")
with open(METRICS_FILE, "w") as f:
    f.write(json.dumps(metrics_dict))

pred_probabilities = model.predict(x_test)
predictions = np.argmax(pred_probabilities, axis=1)
all_predictions = [{"actual": int(actual), "predicted": int(predicted)} for actual, predicted in zip(labels, predictions)]
with open("output/predictions.json", "w") as f:
    json.dump(all_predictions, f)

def add_margin(pil_img, top, right, bottom, left):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), 0)
    result.paste(pil_img, (left, top))
    return result

for n, image in enumerate(raw_x_test[:20]):
    font = ImageFont.load_default()

    text = f"Ground Truth: {labels[n]}\nPrediction: {predictions[n]}"

    # get text size
    text_size = font.getsize(text)

    # set button size + 10px margins
    label_size = (text_size[0]+30, text_size[1]+30)

    label_img = Image.new('RGBA', label_size, "black")
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((10, 10), text, font=font)

    print(image.shape)
    image_pil = Image.fromarray(np.squeeze(image))
    image_pil = add_margin(image_pil, 50, 50, 50, 50)
    image_pil.paste(label_img)

    live.log(f"prediction_{n}.png", image_pil)
