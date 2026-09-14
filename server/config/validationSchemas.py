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
    "markerDetector": {
        "required": False,
        "type": "string",
        "allowed": ["apriltag", "aruco"],
        "default": "apriltag",
    },
    "markerDetectorParams": {
        "required": False,
        "type": "dict",
        "schema": {
            "apriltag": {
                "required": False,
                "type": "dict",
                "schema": {
                    "threads": {"required": False, "type": "integer", "nullable": True, "min": 1},
                    "maxhamming": {"required": False, "type": "integer", "min": 0, "max": 3},
                    "decimate": {"required": False, "type": "float", "min": 0.5, "max": 8.0},
                    "blur": {"required": False, "type": "float", "min": 0.0, "max": 4.0},
                    "refine_edges": {"required": False, "type": "boolean"},
                },
            },
            "aruco": {
                "required": False,
                "type": "dict",
                "schema": {
                    "detectInvertedMarker": {"required": False, "type": "boolean"},
                    "cornerRefinementMethod": {
                        "required": False,
                        "type": "string",
                        "allowed": ["none", "subpix", "contour", "apriltag"],
                    },
                    "adaptiveThreshWinSizeMin": {"required": False, "type": "integer", "min": 3},
                    "adaptiveThreshWinSizeMax": {"required": False, "type": "integer", "min": 3},
                    "adaptiveThreshWinSizeStep": {"required": False, "type": "integer", "min": 1},
                    "adaptiveThreshConstant": {"required": False, "type": "float", "min": 0.0},
                    "minMarkerPerimeterRate": {"required": False, "type": "float", "min": 0.0},
                    "maxMarkerPerimeterRate": {"required": False, "type": "float", "min": 0.0},
                    "polygonalApproxAccuracyRate": {"required": False, "type": "float", "min": 0.0},
                    "cornerRefinementWinSize": {"required": False, "type": "integer", "min": 1},
                    "cornerRefinementMaxIterations": {"required": False, "type": "integer", "min": 1},
                    "cornerRefinementMinAccuracy": {"required": False, "type": "float", "min": 0.0},
                    "errorCorrectionRate": {"required": False, "type": "float", "min": 0.0, "max": 1.0},
                    "aprilTagQuadDecimate": {"required": False, "type": "float", "min": 0.0},
                    "aprilTagQuadSigma": {"required": False, "type": "float", "min": 0.0},
                },
            },
        },
    },
    "mapSegmentation": {
        "required": False,
        "type": "dict",
        "schema": {
            "strategy": {
                "required": False,
                "type": "string",
                "allowed": ["classical", "unet"],
                "default": "classical",
            },
            "classical": {
                "required": False,
                "type": "dict",
                "nullable": True,
            },
            "unet": {
                "required": False,
                "type": "dict",
                "schema": {
                    "modelPath": {"required": False, "type": "string"},
                    "architecture": {
                        "required": False,
                        "type": "string",
                        "allowed": ["multihead", "singlehead"],
                    },
                    "inputSize": {"required": False, "type": "integer", "min": 64, "max": 2048},
                    "device": {
                        "required": False,
                        "type": "string",
                        "allowed": ["auto", "cpu", "cuda", "mps"],
                    },
                    "halfPrecision": {
                        "required": False,
                        "anyof": [
                            {"type": "boolean"},
                            {"type": "string", "allowed": ["auto"]},
                        ],
                    },
                    "maskMarkingsToRoads": {"required": False, "type": "boolean"},
                },
            },
        },
    },
    "useEpipolarGeometry": {"required": False, "type": "boolean", "default": True},
    "usePerCameraSolver": {"required": False, "type": "boolean", "default": False},
    "topDownFitUseInteriorTags": {"required": False, "type": "boolean", "default": True},
    "useAtlasCornerResolution": {"required": False, "type": "boolean", "default": True},
    "topDownStrategy": {"required": False, "type": "string", "allowed": ["mosaic", "orthorectified"], "default": "mosaic"},
    "autoCalibrateOnStart": {"required": False, "type": "boolean", "nullable": True},
}
