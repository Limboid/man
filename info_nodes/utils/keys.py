LOSS = 'loss'

LATENT = 'latent'
LATENT_UNCERTAINTY = 'latent_uncertainty'

PRED_LATENT = 'pred_latent'
PRED_LATENT_UNCERTAINTY = 'pred_latent_uncertainty'

TARGET_LATENTS = 'target_latents'
# storing the list of targets rather than one mixture would result in double computation
# if the targets were distributions, but since I will be drawing random samples from the
# targets, they must all be present.
