import discord
import numpy as np
import re
import tensorflow as tf
from chefbotModel import MainSubclassPrediction
from pythainlp.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from pythainlp import word_vector
from sklearn.preprocessing import OneHotEncoder

# ====================================================================================

def map_word_to_vector(word):
    global wordVector
    try:
        return wordVector[word]
    except KeyError:
        return np.zeros(wordVector.vector_size)
    
def preprocessText(text):
    # Remove newline, white space, mentions, emojis
    text = text.replace("\n", "")
    text = text.replace(" ", "")
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r':\w+:', '', text)

    # Remove special characters
    special = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    thai_special = 'ๆฯ๐ฺ'
    text = text.translate(str.maketrans('', '', special + thai_special))

    # Tokenize text
    text = np.array(word_tokenize(text, engine='newmm'), dtype=object)

    # Padding text
    maxlen = 50
    text = np.pad(text, (0, maxlen - len(text)), constant_values=" ")

    # Vectorize text
    text = np.array([map_word_to_vector(word) for word in text])

    # Reshape text
    text = np.expand_dims(text, axis=0)

    # Convert Numpy array to Tensor
    text = tf.convert_to_tensor(text, dtype=tf.float32)

    return text

def preprocessLabel(label):
    global main_class_label
    label_enc = np.zeros((1, len(main_class_label)))
    label_enc[:,main_class_label.index(label)] = 1

    return label_enc

def model_predict_main(text):
    global main_class_label
    pred = model.predict_mainclass(text)
    pred = [main_class_label[idx] for idx in pred]
    pred = ','.join(pred)
    return pred
    
def model_predict_sub(text, label):
    global sub_class_label
    pred_sub = model.predict_interaction(text, label)
    pred_sub = [sub_class_label[idx] for idx in pred_sub]
    pred_sub = ','.join(pred_sub)
    return pred_sub

# ====================================================================================
#   initailize variable

bot = discord.Client()
TOKEN = "BOT TOKEN"

wordVector = word_vector.WordVector(model_name="thai2fit_wv").get_model()
main_class_label = ['พิซซ่า', 'ก๋วยเตี๋ยว', 'สปาเกตตี']
sub_class_label = ['พิซซ่าค็อกเทลกุ้ง', 'พิซซ่ามีทเดอลุกซ์', 'พิซซ่าเห็ดและมะเขือเทศ', 'พิซซ่าดิปเปอร์', 
            'ก๋วยเตี๋ยวน้ำตก', 'ก๋วยเตี๋ยวต้มยำน้ำใส', 'บะหมีหมูแดงหมูกรอบ', 'เกาเหลา', 
            'สปาเกตตีมีทบอล', 'สปาเกตตีคาโบนาร่า', 'สปาเกตตีผัก', 'สปาเกตตีทะเล']

model = MainSubclassPrediction(50, 300)
model.load_weight_mainclass_inference_model("model_weigth/MainModel.h5")
model.load_weight_interaction_model("model_weigth/InteractionModel.h5")

bot_state = 0
input1 = ""
input2 = ""
main_label = ""
sub_label = ""
start_message = """ยินดีต้อนรับสู้ ChefBot ผู้ช่วยในการคิดมื้ออาหารสำหรับคุณ
โดยคุณสามารถเรียก <@1103259121169473556> และบรรยายข้อมูลที่เกี่ยวกับอาหารได้ในหมวดหมู่ดังนี้
1.รสชาติ
2.รสสัมผัส
3.กรรมวิธีหรือขั้นตอนการทำ
4.รูปลักษณ์
5.ธรรมเนียมหรือวิธีการกิน
6.สารอาหาร
7.ส่วนผสม
Note: /start เพื่อเริ่มการทำงานใหม่"""
reset_message = "Bot has been reset successfully."

@bot.event
async def on_ready():
    print("Bot Started")
    
# ====================================================================================

@bot.event
async def on_message(message):
    global bot_state, input1, input2, main_label, sub_label, start_message, reset_message

    print(message.content)

    if bot_state == 0: # Clear
        print(f"test {bot_state}") 
        if message.author == bot.user:
            return
        
        if bot.user.mentioned_in(message): # Clear
            print(f"content {bot_state}")
            input1 = message.content
            input1 = input1.replace("<@1103259121169473556>", '')
            input1 = input1.replace(' ', '')
            input1 = input1.lower()

            if input1 == "/start": # Clear
                bot_state = 0
                input1 = ""
                main_label = ""
                await message.channel.send(reset_message)
                await message.channel.send(start_message)     

            elif input1 == "": # Clear
                await message.channel.send(start_message)

            else: # Clear
                bot_state = 1
                input1 = preprocessText(input1)
                main_label = model_predict_main(input1)
                output1_message = f"""เมนูที่ต้องการน่าจะอยู่ในหมวดหมู่ {main_label} ใช่หรือไม่ โดยสามารถตอบกลับด้วย <@1103259121169473556> ตามด้วยเลขได้ดังนี้
1 --> หากเมนูที่ึคุณต้องการถูกต้องและแสดงผลลัพท์
2 --> หากเมนูที่ึคุณต้องการถูกต้องและต้องการที่จะให้รายละเอียดเพิ่ม
3 --> หากเมนูไม่ใช่เมนูที่คุณต้องการ"""
                await message.channel.send(output1_message)

    elif bot_state == 1: # Clear
        print(f"test {bot_state}")
        if message.author == bot.user:
            return 
            
        if bot.user.mentioned_in(message): # Clear
            print(f"content {bot_state}")
            text = message.content
            text = text.replace("<@1103259121169473556>", '')
            text = text.replace(' ', '')
            text = text.lower()

            if text == "/start": # Clear
                bot_state = 0
                input1 = ""
                main_label = ""
                await message.channel.send(reset_message) 
                await message.channel.send(start_message)

            elif text == "1": # Clear # หากเมนูที่คุณต้องการถูกต้องและแสดงผลลัพท์
                bot_state = 0
                main_label_encoded = preprocessLabel(main_label)
                sub_label = model_predict_sub(input1, main_label_encoded)
                output2_message = f"เมนูที่ต้องการน่าจะคือ {sub_label}"
                await message.channel.send(output2_message)

            elif text == "2": # Clear # หากเมนูที่คุณต้องการถูกต้องและต้องการที่จะให้รายละเอียดเพิ่ม
                bot_state = 2
                await message.channel.send(f"กรุณาให้รายละเอียดเพิ่มของ {main_label}")

            elif text == "3": # Clear # หากเมนูไม่ใช่เมนูที่คุณต้องการ
                bot_state = 0
                input1 = ""
                main_label = ""
                await message.channel.send("เมนูนี้ไม่ใช่เมนูที่คุณต้องการ")
                await message.channel.send(start_message)

            else: # Clear
                await message.channel.send("กรุณาตอบกลับเป็นเลข 1, 2, 3 หรือ /start เพื่อเริ่มการทำงานใหม่")

    elif bot_state == 2: # Clear
        print(f"test {bot_state}")
        if message.author == bot.user:   
            return         
        if bot.user.mentioned_in(message): # Clear
            print(f"content {bot_state}")
            input2 = message.content
            input2 = input2.replace("<@1103259121169473556>", '')
            input2 = input2.replace(' ', '')
            input2 = input2.lower()

            if input2 == "/start": # Clear
                bot_state = 0
                input1 = ""
                input2 = ""
                main_label = ""
                await message.channel.send(reset_message) 
                await message.channel.send(start_message)

            elif input2 == "": # Clear
                await message.channel.send("กรุณาให้ข้อมูลใหม่ หรือ /start เพื่อเริ่มการทำงานใหม่")
                
            else: # Clear
                bot_state = 0
                input2 = preprocessText(input2)
                main_label_encoded = preprocessLabel(main_label)
                sub_label = model_predict_sub(input1 + input2, main_label_encoded)
                output2_message = f"เมนูที่ต้องการน่าจะคือ {sub_label}"    
                await message.channel.send(output2_message)

# ====================================================================================

if __name__ == '__main__':
    bot.run(TOKEN)