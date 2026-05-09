logLevelSchema = {
    "required": True,
    "type": "string",
    "allowed": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
}

robotKalmanFilterSchema = {
    "enabled": {"required": False, "type": "boolean"},
    "gracePeriodRounds": {"required": False, "type": "integer", "min": 0},
    "maxSpeedNormPerSec": {"required": False, "type": "float", "min": 1e-9},
    "maxYawRateDegPerSec": {"required": False, "type": "float", "min": 1e-9},
    "processNoiseLinearVelocityStd": {"required": False, "type": "float", "min": 1e-12},
    "processNoiseAngularRateStdDeg": {"required": False, "type": "float", "min": 1e-12},
    "positionSigmaMin": {"required": False, "type": "float", "min": 1e-12},
    "positionSigmaMax": {"required": False, "type": "float", "min": 1e-12},
    "azimuthSigmaMinDeg": {"required": False, "type": "float", "min": 1e-12},
    "azimuthSigmaMaxDeg": {"required": False, "type": "float", "min": 1e-12},
    "positionSigmaFromAreaCoefficient": {"required": False, "type": "float", "min": 1e-12},
    "azimuthSigmaFromBaselineCoefficientDeg": {"required": False, "type": "float", "min": 1e-12},
}

configSchema = {
    "debug": {
        "required": True,
        "type": "boolean",
    },
    "profiling": {
        "required": True,
        "type": "boolean",
    },
    "algorithmLogLevel": logLevelSchema,
    "algorithmInstrumenterLogLevel": logLevelSchema,
    "serverLogLevel": logLevelSchema,
    "httpLogLevel": logLevelSchema,
    "interThreadMemoryLogLevel": logLevelSchema,
    "otherLogLevel": logLevelSchema,
    "algorithmJPEGQuality": {"required": True, "type": "integer", "min": 0, "max": 100},
    "algorithmWaitForHTTPServerReady": {
        "required": False,
        "type": "boolean",
        "default": True,
    },
    "recalcHeuristicExtinguishingMarkerConsecutiveRoundsDelay": {
        "required": False,
        "type": "integer",
        "min": 0,
        "default": 4,
    },
    "recalcHeuristicAppearingMarkerConsecutiveRoundsDelay": {
        "required": False,
        "type": "integer",
        "min": 0,
        "default": 4,
    },
    "markerVisibilityEwmaAlpha": {
        "required": False,
        "type": "float",
        "min": 0.01,
        "max": 0.99,
        "default": 0.35,
    },
    "markerVisibilityThresholdHigh": {
        "required": False,
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.75,
    },
    "markerVisibilityThresholdLow": {
        "required": False,
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.25,
    },
    "caching": {
        "required": False,
        "type": "dict",
        "schema": {
            "enabled": {"required": False, "type": "boolean", "default": True},
            "stitchingHomographyCache": {
                "required": False,
                "type": "boolean",
                "default": True,
            },
            "topDownHomographyCache": {
                "required": False,
                "type": "boolean",
                "default": True,
            },
            "mapSegmentationThrottle": {
                "required": False,
                "type": "boolean",
                "default": True,
            },
            "mapSegmentationIntervalSeconds": {
                "required": False,
                "type": "integer",
                "min": 0,
                "default": 3,
            },
        },
    },
    "preallocation": {
        "required": False,
        "type": "dict",
        "schema": {
            "adaptiveBufferPreallocation": {
                "required": False,
                "type": "boolean",
                "default": True,
            },
            "homographyBufferPreallocation": {
                "required": False,
                "type": "boolean",
                "default": True,
            },
        },
    },
    "http": {
        "type": "dict",
        "schema": {
            "host": {"required": True, "type": "string"},
            "port": {
                "required": True,
                "type": "integer",
                "min": 0,
                "max": 65535,
            },
        },
    },
    "map": {
        "required": True,
        "type": "dict",
        "schema": {
            key: {
                "required": True,
                "type": "integer",
            }
            for key in ["TL", "TR", "BL", "BR"]
        },
    },
    "dimensions": {
        "required": True,
        "type": "dict",
        "schema": {
            key: {
                "required": True,
                "type": "integer",
            }
            for key in ["mapWidth", "mapHeight", "markerWidth", "markerHeight"]
        },
    },
    "robotKalmanFilter": {
        "required": False,
        "type": "dict",
        "schema": robotKalmanFilterSchema,
    },
    "robots": {
        "required": True,
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "id": {
                    "required": True,
                    "type": "integer",
                    # AprilTag 36h11 limits
                    "min": 0,
                    "max": 586,
                },
                "host": {"required": True, "type": "string"},
                "kalman": {
                    "required": False,
                    "type": "dict",
                    "schema": robotKalmanFilterSchema,
                },
            },
        },
    },
    "rtspSources": {
        "required": False,
        "type": "list",
        "schema": {
            "type": "string",
        },
    },
    "playbackModeLoop": {"required": True, "type": "boolean"},
}
