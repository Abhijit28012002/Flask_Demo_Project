from PIL import Image
import numpy as np

def encode_image(original_image, secret_message,path):
    img = Image.open(original_image)
    pixels = np.array(img)

    message_binary = ''.join(format(ord(char), '08b') for char in secret_message)
    message_binary += '1111111111111110'  # Add a delimiter to mark the end of the message

    index = 0
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            for color_channel in range(3):  # R, G, B
                if index < len(message_binary):
                    pixels[i][j][color_channel] = int(format(pixels[i][j][color_channel], '08b')[:-1] + message_binary[index], 2)
                    index += 1

    encoded_image = Image.fromarray(pixels.astype('uint8'))
    encoded_image.save(f'./static/images/Download/{path}')

def decode_image(encoded_image):
    img = Image.open(encoded_image)
    pixels = np.array(img)

    binary_data = ''
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            for color_channel in range(3):  # R, G, B
                binary_data += format(pixels[i][j][color_channel], '08b')[-1]

    message = ''
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i+8]
        if byte == '11111111':
            break
        else:
            message += chr(int(byte, 2))

    return message


# Example for image steganography
#encode_image('./static/images/Download/Intel Uses Python.jpg', 'Hello, this is a secret message!','Intel.png')
#decoded_message_image = decode_image('./static/images/Download/Intel.png')
#print(f"Decoded Image Message: {decoded_message_image}")

# Example for video steganography
#encode_video('original_video.mp4', 'Hello, this is a secret message!')
#decoded_message_video = decode_video('encoded_video.avi')
#print(f"Decoded Video Message: {decoded_message_video}")
