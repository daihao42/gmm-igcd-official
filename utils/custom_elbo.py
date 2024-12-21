from numpyro.infer import SVI, Trace_ELBO
import numpyro
import jax.numpy as jnp
import jax

from numpyro.infer.util import log_density, get_importance_trace


# Define a custom ELBO function

class CustomELBO(Trace_ELBO):
    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        # Compute the standard ELBO loss
        elbo_loss = super().loss(rng_key, param_map, model, guide, *args, **kwargs)


        # Extract log_density outputs
        #model_log_prob_sum, model_site_log_probs = log_density(model, args, kwargs, param_map)
        #guide_log_prob_sum, guide_site_log_probs = log_density(guide, args, kwargs, param_map)
        # Compute ELBO
        #elbo_loss_test = model_log_prob_sum - guide_log_prob_sum

        #print("ELBO loss: ", elbo_loss_test, elbo_loss)



        previous_covs_dets = kwargs.get("covs_dets", None)

        dets_loss = ((previous_covs_dets[0] - previous_covs_dets[1]) / previous_covs_dets[0]) ** 2

        dets_loss = dets_loss.item()
        
        # Example: Add a regularization term (e.g., L2 norm on parameters)
        reg_term = 0.0
        for param_name, param_value in param_map.items():
            reg_term += jnp.sum(param_value ** 2)  # L2 regularization

        # Combine the original ELBO with the regularization term
        custom_loss = elbo_loss + dets_loss + reg_term  # Add a weight for the regularization term
        return custom_loss
