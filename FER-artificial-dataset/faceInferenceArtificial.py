import torch
from diffusers import StableDiffusionPipeline

torch.cuda.empty_cache()
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)

pipe = pipe.to("cuda")

import time

gender = {1:'male', 2:'female'}
age = {1:'child',2:'young',3:'adult',4:'old'}
ethnicity = {1:'asian',2:'black',3:'indian',4:'latinamerican',5:'white'}
category = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}

g = input('Gender ' + str(gender) + '? ')
#print(gender[int(g)]) 
a = input('Age ' + str(age) + '? ')
#print(age[int(a)])
e = input('Ethnia ' + str(ethnicity) + '? ')
#print(ethnicity[int(e)])
#p = input('Category? ')
n = input('Images? ')

prompt_angry = "A detailed photographic portrait of a perfect face of a " + ethnicity[int(e)] + " " + age[int(a)] + " " + gender[int(g)] + ", \
feeling an extreme rage, \
expressing a very angry face, \
features well-defined, \
facing the camera, realistic, 4K, hd"

# forehead wrinkler, \
# brow lowerer, \
# upper lid raiser, \
# lid tightener, \
# upper lip raiser, \
# lip tightener, \
# lip part, \

prompt_disgust = "A detailed photographic portrait of a perfect face of a " + ethnicity[int(e)] + " " + age[int(a)] + " " + gender[int(g)] + ", \
feeling angry, \
with expression of very disgusted, \
forehead wrinkler, \
brow lowerer, \
cheek raiser, \
narrowed eyes, \
nose wrinkler, \
upper lip raiser, \
chin raiser, \
lip part, \
facing the camera, realistic, 4K, hd"

prompt_fear = "A detailed photographic portrait of a perfect face of a " + ethnicity[int(e)] + " " + age[int(a)] + " " + gender[int(g)] + ", \
expressing great dread, \
with facial gestures of fearful, fright, in panic, \
facing the camera, realistic, 4K, hd"
#inner brow raiser, \outer brow raiser, \brow lowerer, \upper lid raiser, \upper lip raiser, \lip part, \jaw drop, \

prompt_happy = "A detailed photo portrait of a perfect whole front face of a " + ethnicity[int(e)] + " " + age[int(a)] + " " + gender[int(g)] + ", \
expressing very happiness, \
facial gestures of happy, smiling, \
facial features well-defined, facing the camera, background any, ultra realistic, 4K, hd"

prompt_neutral = "A detailed photo portrait of a perfect face of a " + ethnicity[int(e)] + " " + age[int(a)] + " " + gender[int(g)] + ", \
expressing neutrality, \
facial gestures of neutral, \
facial features well-defined, facing the camera, background any, ultra realistic, 4K, hd"

prompt_sad = "A detailed photo portrait of a perfect whole front face of a " + ethnicity[int(e)] + " " + age[int(a)] + " " + gender[int(g)] + ", \
very sad face with tears, \
expressing extreme frustration, \
crying, \
facial features well-defined, facing the camera, background any, ultra realistic, 4K, hd"

prompt_surprise = "A detailed photographic portrait of a perfect face of a " + ethnicity[int(e)] + " " + age[int(a)] + " " + gender[int(g)] + ", \
expressive face of surprise with the mouth open, \
extremely amazed, \
eyebrows very raised, \
upper eyelid raised, \
lips parted, \
jaw dropped, \
facial features well-defined, facing the camera, background any, realistic, 4K, hd"

print(prompt_fear)
i=0
for i in range(int(n)):
    print(str(i))
    image = pipe(prompt_fear).images[0]
    #image.save("angry-" + ethnicity[int(e)] + "-" + age[int(a)] + "-" + gender[int(g)] + "-" + str(time.strftime("%H_%M_%S")) + ".png")
    #image.save("disgust-" + ethnicity[int(e)] + "-" + age[int(a)] + "-" + gender[int(g)] + "-" + str(time.strftime("%H_%M_%S")) + ".png")
    image.save("fear-" + ethnicity[int(e)] + "-" + age[int(a)] + "-" + gender[int(g)] + "-" + str(time.strftime("%H_%M_%S")) + ".png")
    #image.save("happy-" + ethnicity[int(e)] + "-" + age[int(a)] + "-" + gender[int(g)] + "-" + str(time.strftime("%H_%M_%S")) + ".png")
    #image.save("neutral-" + ethnicity[int(e)] + "-" + age[int(a)] + "-" + gender[int(g)] + "-" + str(time.strftime("%H_%M_%S")) + ".png")
    #image.save("sad-" + ethnicity[int(e)] + "-" + age[int(a)] + "-" + gender[int(g)] + "-" + str(time.strftime("%H_%M_%S")) + ".png")
    #image.save("surprise-" + ethnicity[int(e)] + "-" + age[int(a)] + "-" + gender[int(g)] + "-" + str(time.strftime("%H_%M_%S")) + ".png")