import cv2

def text_to_binary(text):
    binary_message = ''.join(format(ord(char), '08b') for char in text)
    return binary_message

def encode_text(video_path, secret_text, output_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    binary_secret_text = text_to_binary(secret_text)
    binary_secret_text += "1111111111111110"  # Add a delimiter to mark the end of the message

    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()

    cap.release()

    height, width, _ = frames[0].shape

    index = 0
    for i in range(height):
        for j in range(width):
            for color_channel in range(3):  # Iterate through RGB channels
                if index < len(binary_secret_text):
                    # Modify the LSB of the pixel's color channel
                    frames[0][i][j][color_channel] &= ~1
                    frames[0][i][j][color_channel] |= int(binary_secret_text[index])
                    index += 1

    cv2.imwrite(output_path, frames[0])

def decode_text(encoded_image_path):
    encoded_image = cv2.imread(encoded_image_path)

    binary_secret_text = ''
    for i in range(encoded_image.shape[0]):
        for j in range(encoded_image.shape[1]):
            for color_channel in range(3):
                # Extract the LSB of each color channel to reconstruct the binary message
                binary_secret_text += str(encoded_image[i][j][color_channel] & 1)

    # Find the delimiter and extract the original text
    delimiter_index = binary_secret_text.find('1111111111111110')
    original_binary_text = binary_secret_text[:delimiter_index]
    original_text = ''.join([chr(int(original_binary_text[i:i+8], 2)) for i in range(0, len(original_binary_text), 8)])

    return original_text

# Example usage:
#video_path = './static/newVideos.mp4'
#secret_text = 'Hello, this is a secret message!'
#output_path = './static/output.png'

# Encode the text into the video
#encode_text(video_path, secret_text, output_path)

# Decode the text from the encoded video
#decoded_text = decode_text(output_path)
#print("Decoded Text:", decoded_text)
