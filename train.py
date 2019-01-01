import input_pipeline
import MyKeras
import models

BDD100K = "H:/Dataset/bdd100k"
BATCH_SIZE = 8

train_gen, valid_gen = input_pipeline.get_generators(batch_size=BATCH_SIZE, dataset_path=BDD100K)

load = MyKeras.ask_load()

if load:
    model, epoch = MyKeras.load_latest_model('main')
else:
    model = models.build_model()
    epoch = 0

model = MyKeras.fit_model(model, valid_gen, valid_gen, initial_epoch=epoch,
                          train_steps=None, valid_steps=100,
                          model_save_dir='models/main', log_dir='logs/main')
