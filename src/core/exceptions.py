class ProjectError(Exception):
    pass


class ConfigurationError(ProjectError):
    pass


class DeviceError(ProjectError):
    pass


class ModelLoadError(ProjectError):
    pass


class QuantizationError(ProjectError):
    pass


class StreamingError(ProjectError):
    pass


class InferenceError(ProjectError):
    pass


class BenchmarkError(ProjectError):
    pass


class MPIError(ProjectError):
    pass