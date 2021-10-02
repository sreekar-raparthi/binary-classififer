from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from generator import Generator
from model import Model
import argparse
import h5py

width, height = 75,75

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--image_dir", type=str, default="D:/Data/", help="image") 
    parser.add_argument("--filename_label_train", type=str, default="D:/train_triple_class.json", help="json file including all training filenames")
    parser.add_argument("--filename_label_test", type=str, default="D:/validation_triple_class.json", help="json file including all validation filenames")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()

    train_generator = Generator(image_dir=args.image_dir, label_json=args.filename_label_train, batch_size=args.batch_size)     
    validation_generator = Generator(image_dir=args.image_dir, label_json=args.filename_label_test, batch_size=args.batch_size)

    model = Model(width,height)
    model = model.build_model()

    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                optimizer = adam,
                metrics=['accuracy'])

    training_steps = train_generator.total_samples / args.batch_size
    validation_steps=validation_generator.total_samples / args.batch_size
    
    
    checkpointer = ModelCheckpoint('model_best_classifier.h5', verbose=1, save_best_only=True)
    tb_cb = TensorBoard(log_dir="tb_log/")
    
    history =  model.fit_generator(generator=train_generator.flow(),
                        steps_per_epoch=training_steps,
                        epochs=args.num_epochs,callbacks = [checkpointer,tb_cb],
                        validation_data=validation_generator.flow(),
                        validation_steps= validation_steps)
