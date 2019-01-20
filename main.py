from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"




# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
# myGenerator = trainGenerator(8,'data/bottles/train','image','label',data_gen_args,save_to_dir = "data/bottles/train/aug")
#
# num_batch = 3
# for i,batch in enumerate(myGenerator):
#     if(i >= num_batch):
#         break
#
# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
#
# myGene = trainGenerator(2,'data/bottles/train','image','label',data_gen_args,save_to_dir = None)
# myGene_mem = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)

model = unet()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit_generator(myGene, steps_per_epoch=50, epochs=5, callbacks=[model_checkpoint])
# model.fit_generator(myGene_mem,steps_per_epoch=50,epochs=5,callbacks=[model_checkpoint])

model.load_weights("unet_membrane.hdf5")
testGene = testGenerator("data/bottles/test")
# testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,3,verbose=1)

saveResult("data/bottles/test",results)
# saveResult("data/membrane/test",results)