"""ReMDM-cap + switch, adapted to the pinned Task 1 ancestral sampler.

Equations 6,7 of https://arxiv.org/abs/2503.00307. No confidence heuristic,
loop schedule, nucleus sampling, or change to model/categorical precision.
Visible tokens remain fixed OR become mask, then can change on a later step.
"""
import torch

from samplers import AncestralSampler, AncestralContext, sample_categorical


def transition_probabilities(alpha_t, alpha_s, eta):
    """Return P(visible -> mask), P(mask -> token); alpha_t < 1."""
    bound = ((1 - alpha_s) / alpha_t.clamp_min(1e-30)).clamp(0, 1)
    remask = torch.minimum(torch.full_like(bound, eta), bound)
    reveal = (alpha_s - (1 - remask) * alpha_t) / (1 - alpha_t)
    return remask, reveal.clamp(0, 1)


class ReMDMSampler(AncestralSampler):
    def __init__(self, *, eta, switch=.5, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= eta <= 1 or not 0 < switch <= 1:
            raise ValueError('Invalid eta or switch')
        self.eta = eta
        self.switch = switch

    def init_state(self, model, num_samples, **kwargs):
        state = super().init_state(model, num_samples, **kwargs)
        assert model.diffusion_type == 'absorbing'
        state.mask_index = model.mask_index
        assert self.noise_removal == 'ancestral' and self.steps_policy == 'full'
        state.editable = (torch.arange(state.xt.shape[1], device=state.xt.device)[None, :]
                          >= state.prefix_lengths[:, None]) if state.prefix_lengths is not None else torch.ones_like(state.xt, dtype=torch.bool)
        state.last_visible = state.xt.clone()
        state.changed_positions = torch.zeros_like(state.xt, dtype=torch.bool)
        # Counters remain on device until metadata, avoiding stepwise CPU sync.
        state.remask_events = torch.zeros(num_samples, device=state.xt.device, dtype=torch.long)
        state.rereveal_events = torch.zeros_like(state.remask_events)
        state.changed_events = torch.zeros_like(state.remask_events)
        state.prompt_violations = torch.zeros_like(state.remask_events)
        state.active_steps = 0
        # Evaluate switch on CPU once; do not synchronize CUDA each step.
        state.active_indices = set(i for i, t in enumerate(state.timesteps.detach().cpu().tolist()[:-1])
                                   if self.eta > 0 and t <= self.switch and i < len(state.timesteps)-2)
        return state

    def step(self, model, state):
        before = state.xt.clone()
        if state.step_idx not in state.active_indices:
            # Native path gives bitwise-equivalent eta=0, including RNG usage.
            state = super().step(model, state)
        else:
            t = state.timesteps[state.step_idx] * state.ones
            s = state.timesteps[state.step_idx + 1] * state.ones
            _, alpha_t = model.noise(t)
            _, alpha_s = model.noise(s)
            log_probs = model.forward(xt=state.xt, sigma=model._sigma_from_alphat(alpha_t),
                                     context=AncestralContext(temperature=self.temperature))
            if self.use_float64:
                log_probs = log_probs.to(torch.float64)
            xt = state.xt[:, state.start_idx:]
            eligible = state.editable[:, state.start_idx:]
            x0 = sample_categorical(log_probs[:, state.start_idx:].exp())
            remask, reveal = transition_probabilities(alpha_t, alpha_s, self.eta)
            was_mask = xt == model.mask_index
            should_reveal = torch.rand_like(x0, dtype=torch.float64) < reveal
            should_remask = (torch.rand_like(x0, dtype=torch.float64) < remask) & ~was_mask & eligible
            # Crucial: only previously masked positions can reveal this step.
            new = torch.where(was_mask & should_reveal & eligible, x0, xt)
            xt[:] = torch.where(should_remask, model.mask_index, new)
            # Measure pre-projection prompt invariance, not only the final output.
            state.prompt_violations += ((before != state.xt) & ~state.editable).sum(-1)
            self._project_prefix(state.xt, state.prefix_tokens, state.prefix_lengths)
            state.nfe += 1
            state.step_idx += 1
            state.active_steps += 1
        after = state.xt
        remasked = (before != model.mask_index) & (after == model.mask_index) & state.editable
        rerevealed = ((before == model.mask_index) & (after != model.mask_index)
                      & (state.last_visible != model.mask_index) & state.editable)
        changed = rerevealed & (after != state.last_visible)
        state.remask_events += remasked.sum(-1)
        state.rereveal_events += rerevealed.sum(-1)
        state.changed_events += changed.sum(-1)
        state.changed_positions |= changed
        state.last_visible = torch.where(after != model.mask_index, after, state.last_visible)
        return state

    def metadata(self, state):
        return {'nfe': state.nfe, 'eta': self.eta, 'switch': self.switch,
                'active_steps': state.active_steps,
                'remask_events': state.remask_events.cpu().tolist(),
                'rereveal_events': state.rereveal_events.cpu().tolist(),
                'changed_events': state.changed_events.cpu().tolist(),
                'changed_positions': state.changed_positions.sum(-1).cpu().tolist(),
                'prompt_violations': state.prompt_violations.cpu().tolist(),
                'remaining_masks': ((state.xt == state.mask_index) & state.editable).sum(-1).cpu().tolist()}
