.. _adaptive_preallocation:

Adaptive NumPy Buffer Preallocation
===================================

This document describes the adaptive preallocation algorithm used by the MCMRVLS server
to manage numpy array buffers for image processing pipelines. The algorithm is inspired
by queuing systems theory and adaptively reallocates buffers by enlarging or shrinking
based on observed demand.

Motivation
----------

Real-time multi-camera map building produces variable-sized outputs: stitched mosaics
and top-down maps depend on camera geometry, homography estimation, and corner marker
detections. Allocating new numpy arrays on every frame incurs allocation overhead and
fragmentation. Preallocating buffers and reusing them reduces GC pressure and improves
throughput. The challenge is that output dimensions vary over time—hence the need for
an *adaptive* allocator.

Queuing-Theory Foundation
-------------------------

The design draws on classical queuing concepts.

Little's Law Analogy
~~~~~~~~~~~~~~~~~~~

In an M/M/1 queue, Little's Law states :math:`L = \lambda W`: the average number of
items in the system equals arrival rate times average sojourn time. For our buffers,
we treat *demand* (required buffer dimensions per frame) as an arrival process and
estimate steady-state capacity via an **exponentially weighted moving average (EWMA)**.

- **EWMA of observed demand**: Let :math:`S_t` be the observed size at time :math:`t`.
  We update :math:`\bar{S} = \alpha S_t + (1-\alpha)\bar{S}` with
  :math:`\alpha \in (0,1)` (e.g. 0.15). This smooths short-term bursts and approximates
  steady-state expected demand.

Growth Policy
~~~~~~~~~~~~

When observed demand exceeds capacity, we **enlarge** with a multiplicative factor
:math:`\gamma > 1` (e.g. 1.25):

.. math::
   C_{\mathrm{new}} = \max\left( S_{\mathrm{needed}} \cdot \gamma,\; C_{\mathrm{current}} \right)

This prevents *starvation* under bursty demand: a single oversized frame does not
force repeated small reallocations.

Shrink Policy (Hysteresis)
~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid *thrashing* (repeated shrink-then-grow cycles), we shrink only when:

1. **Utilization** :math:`u = S_{\mathrm{needed}} / C_{\mathrm{current}}` is below a threshold
   :math:`\tau_u` (e.g. 0.5) for
2. **At least** :math:`K` consecutive rounds (e.g. 10).

This is analogous to a *reservoir drain* or *leaky bucket*: we only release capacity
when demand has been consistently low.

.. math::
   \text{Shrink if} \quad u < \tau_u \quad \text{for} \quad K \quad \text{rounds}

New capacity after shrink:

.. math::
   C_{\mathrm{new}} = \max\left( \lceil \bar{S} \cdot \gamma \rceil,\; S_{\mathrm{needed}},\; C_{\mathrm{min}} \right)

Algorithm Summary
-----------------

1. **Initial allocation**: On first request for size :math:`(h, w)`, allocate
   :math:`(h \gamma, w \gamma)`.

2. **Enlarge**: If :math:`h > C_h` or :math:`w > C_w`, allocate
   :math:`(\max(h\gamma, C_h), \max(w\gamma, C_w))`.

3. **Shrink**: If utilization :math:`u < \tau_u` for :math:`K` rounds, shrink to
   EWMA-based size with safety margin.

4. **Slot delivery**: Return a view :math:`\texttt{buffer}[:h, :w]` into the
   preallocated array. Algorithms write into this view; the allocator owns the
   underlying buffer.

Configuration Parameters
------------------------

+-----------------------------+----------+--------------------------------------------------+
| Parameter                   | Default  | Description                                      |
+=============================+==========+==================================================+
| ``growth_factor``           | 1.25     | Multiplicative factor for enlargements           |
+-----------------------------+----------+--------------------------------------------------+
| ``shrink_utilization_threshold`` | 0.5   | Utilization below which shrink is considered     |
+-----------------------------+----------+--------------------------------------------------+
| ``shrink_consecutive_rounds``| 10      | Consecutive rounds below threshold before shrink |
+-----------------------------+----------+--------------------------------------------------+
| ``ewma_alpha``              | 0.15     | Smoothing factor for demand EWMA                 |
+-----------------------------+----------+--------------------------------------------------+
| ``min_capacity``            | 64       | Minimum dimension to avoid tiny buffers          |
+-----------------------------+----------+--------------------------------------------------+

Buffers Managed
---------------

- **topDownImg**: Top-down warped map image from perspective transform. Dimensions
  depend on corner geometry and aspect ratio.

- **stitchedImg / stitchedImgAnnotated** (future): Stitched mosaic dimensions follow
  input camera resolution.

- **Map segmentation masks** (future): Same size as top-down map.

WebSocket Events
----------------

When homography caches are invalidated or buffers are reallocated, the server emits
WebSocket events to clients subscribed to ``cache_events``. See :ref:`websocket_topics`
for the full data format of all Socket.IO topics.

Implementation
--------------

See :mod:`utils.adaptive_buffer_allocator` for the Python implementation.
