.. _websocket_topics:

WebSocket (Socket.IO) Topics and Data Format
============================================

The server uses Python-SocketIO for real-time WebSocket communication. Clients subscribe to **rooms** by emitting subscription events; the server then emits data only to subscribers.

Connection
---------

Connect to the server (default: ``ws://localhost:6001`` or ``http://localhost:6001``). Use the Socket.IO client library for your platform.

Subscription Events (Client → Server)
-------------------------------------

| Event                       | Room                    | Effect                                  |
|-----------------------------|-------------------------|-----------------------------------------|
| ``subscribe_all_detections``| all_detections          | Receive top-down detections and robots  |
| ``unsubscribe_all_detections`` | all_detections      | Stop receiving                          |
| ``subscribe_algorithm_previews`` | algorithm_previews | Receive stitched/top-down JPEG previews  |
| ``unsubscribe_algorithm_previews`` | algorithm_previews | Stop receiving                    |
| ``subscribe_stitching_previews`` | stitching_previews  | Receive common-features JPEGs per stage |
| ``unsubscribe_stitching_previews`` | stitching_previews | Stop receiving                     |
| ``subscribe_map_numpy_feed``| map_numpy               | Receive raw map segmentation as bytes   |
| ``unsubscribe_map_numpy_feed`` | map_numpy           | Stop receiving                          |
| ``subscribe_cache_events``  | cache_events            | Receive homography/buffer event telemetry|
| ``unsubscribe_cache_events`` | cache_events         | Stop receiving                          |
| ``subscribe_map_segmentation_previews`` | map_segmentation_previews | Receive map mask JPEGs     |
| ``unsubscribe_map_segmentation_previews`` | map_segmentation_previews | Stop receiving           |
| ``subscribe_camera_previews`` | camera_previews      | Receive per-camera annotated JPEGs      |
| ``subscribe_single_robot``  | robot_{id}              | Per-robot detection feed (pass ``robotId``) |
| ``unsubscribe_single_robot`` | robot_{id}            | Stop receiving                           |

Server Events and Data Format (Server → Client)
----------------------------------------------

all_detections
~~~~~~~~~~~~~~

Emitted to ``all_detections`` room when processing completes.

.. code-block:: json

   {
     "topDownDetections": [{"data": 585, "topLeft": [x,y], "topRight": [...], "bottomLeft": [...], "bottomRight": [...], "centroid": [x,y]}],
     "detections": [{"robot": {"id": 500, "host": "d1.local"}, "x": 0.5, "y": 0.3, "azimuth": 90, "raw_x": 0.51, "raw_y": 0.29, "raw_azimuth": 91, "poseSource": "detection"}],
     "fps": 22.5,
     "dataStale": false
   }

- ``topDownDetections``: list of marker DTOs (topLeft, topRight, bottomLeft, bottomRight, centroid, data)
- ``detections``: robot detection DTOs (filtered ``x``, ``y``, ``azimuth``; optional ``raw_x``, ``raw_y``, ``raw_azimuth`` from the same-frame tag measurement when available; ``poseSource``: ``"detection"`` or ``"estimation"`` when the Kalman filter is coasting after a lost tag)
- ``fps``: estimated frames per second
- ``dataStale``: true when the main algorithm loop failed (stitching/processing error); displayed data may be outdated


algorithm_previews
~~~~~~~~~~~~~~~~~~

Emitted to ``algorithm_previews`` room. JPEG-encoded images as raw bytes.

.. code-block:: json

   {
     "topDownImg": "<bytes>",
     "topDownImgAnnotated": "<bytes>",
     "stitchedImg": "<bytes>",
     "stitchedImgAnnotated": "<bytes>"
   }


stitching_previews
~~~~~~~~~~~~~~~~~~

Emitted to ``stitching_previews`` room. Common-features visualization per stitching stage.

.. code-block:: json

   {
     "commonFeaturesJPEGsMap": {
       "0 x 1": "<bytes>",
       "2 x 3": "<bytes>"
     }
   }

Keys are stitching stage descriptions (e.g. camera pair indices).


map_segmentation_previews
~~~~~~~~~~~~~~~~~~~~~~~~~

Emitted to ``map_segmentation_previews`` room. Map masks as JPEG bytes.

.. code-block:: json

   {
     "redStopLines": "<bytes>",
     "yellowLines": "<bytes>",
     "roadComponents": "<bytes>"
   }


map_segmentation_numpy
~~~~~~~~~~~~~~~~~~~~~~

Emitted to ``map_numpy`` room. Raw numpy arrays as bytes plus metadata.

.. code-block:: json

   {
     "metersPerPixel": 0.01,
     "widthPx": 1200,
     "heightPx": 800,
     "roadsMask": "<bytes>",
     "stopLinesMask": "<bytes>",
     "laneSeparatorLinesMask": "<bytes>"
   }

Masks are uint8 grayscale; reshape to ``(heightPx, widthPx)``.


cache_event
~~~~~~~~~~~

Emitted to ``cache_events`` room when homography caches are invalidated or preallocated buffers are reallocated.

**Homography cache invalidated**

.. code-block:: json

   {
     "type": "homography_cache_invalidated",
     "stage": "0 x 1",
     "reason": "Marker 585 - action: appeared",
     "cache_type": "stitching"
   }

- ``stage``: stitching stage (e.g. "0 x 1") or ``"top_down"``
- ``reason``: human-readable invalidation reason
- ``cache_type``: ``"stitching"`` or ``"top_down"``


**Buffer reallocated**

.. code-block:: json

   {
     "type": "buffer_reallocated",
     "allocator_id": "topDownImg",
     "direction": "enlarge",
     "old_shape": [640, 640],
     "new_shape": [800, 800]
   }

- ``allocator_id``: buffer identifier (e.g. ``topDownImg``)
- ``direction``: ``"enlarge"`` or ``"shrink"``
- ``old_shape``: ``[height, width]`` before reallocation
- ``new_shape``: ``[height, width]`` after reallocation


camera_previews
~~~~~~~~~~~~~~~

Emitted to ``camera_previews`` room. Per-camera images with detections drawn.

.. code-block:: json

   {
     "0": "<bytes>",
     "1": "<bytes>"
   }

Keys are camera indices; values are JPEG bytes.


single_robot_results
~~~~~~~~~~~~~~~~~~~

Emitted to ``robot_{id}`` room. Per-robot detection data.

.. code-block:: json

   {
     "robot": "500",
     "detection": {"x": 0.5, "y": 0.3, "azimuth": 45, "poseSource": "detection"},
     "fps": 22.5
   }
