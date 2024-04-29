from .models import *

def build_model(args, class_weights: List[float] = None):

    if args.model == "spine-dino":

        model = SpineDINO(
                 args, 
                 n_classes= args.n_classes, 
                 n_keypoints = args.n_keypoints, 
                 n_dim = args.n_dims, 
                 n_channels = 3,
                 class_weights = class_weights
        )

    else:
        raise NotImplementedError
    
    return model