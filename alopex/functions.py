# flake8: noqa

from alopex._src.functions import absolute_error
from alopex._src.functions import l1_loss

from alopex._src.functions import accuracy
from alopex._src.functions import accuracy_with_integer_labels
from alopex._src.functions import permutate
from alopex._src.functions import reverse_grad

# Functions from optax.
from optax import smooth_labels
from optax import squared_error
from optax import l2_loss
from optax import log_cosh
from optax import sigmoid_binary_cross_entropy
from optax import softmax_cross_entropy
from optax import softmax_cross_entropy_with_integer_labels
from optax import convex_kl_divergence
from optax import kl_divergence
from optax._src.loss import kl_divergence_with_log_targets
from optax import cosine_similarity
from optax import cosine_distance
from optax import log_cosh
from optax import ctc_loss
from optax import ctc_loss_with_forward_probs
from optax import hinge_loss
from optax import huber_loss
