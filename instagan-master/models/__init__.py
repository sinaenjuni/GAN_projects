import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model()
    instance.initialize(opt)
    print("model [%s] was created" % (instance.name()))
    return instance



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='insta_gan',
                        help='chooses which model to use. cycle_gan, pix2pix, test')
    parser.add_argument('--gpu_ids', type=list, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--ins_max', type=int, default=4, help='maximum number of instances to forward')
    parser.add_argument('--ins_per', type=int, default=2, help='number of instances to forward, for one pass')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')

    parser.add_argument('--netG', type=str, default='set', help='selects model to use for netG')
    parser.add_argument('--netD', type=str, default='set', help='selects model to use for netD')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='ndirectionetwork initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_lsgan', action='store_true',
                        help='do *not* use least square GAN, if false, use vanilla GAN')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')

    parser.add_argument('--pool_size', type=int, default=50,
                        help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr_policy', type=str, default='lambda',
                        help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

    parser.add_argument('--loadSizeW', type=int, default=220, help='scale images to this size (width)')
    parser.add_argument('--loadSizeH', type=int, default=220, help='scale images to this size (height)')
    parser.add_argument('--fineSizeW', type=int, default=200, help='then crop to this size (width)')
    parser.add_argument('--fineSizeH', type=int, default=200, help='then crop to this size (height)')
    parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                        help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
    parser.add_argument('--no_flip', action='store_true',
                        help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--serial_batches', action='store_true', default=False,
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

    args = parser.parse_args(args=[])
    args.isTrain = 'train'
    args.gpu_ids = [0]

    model = create_model(args)
    model.setup(args)
    print(model.netG_A)
    print(model.net)