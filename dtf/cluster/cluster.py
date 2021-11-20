import tensorflow as tf

class Cluster:
    def __init__(self, cluster_dict, name, task):
        self.name = name
        self.task = task
        self._server = None
        self._cluster_dict = cluster_dict
        self._spec = tf.train.ClusterSpec(cluster_dict)

    def connect(self):
        tf.config.experimental_connect_to_cluster(self._spec)
        devices = tf.config.list_logical_devices()
        print("Connected to cluster with devices:")
        for d in devices:
            print("\t" + d.name)

    def make_server(self):
        return tf.distribute.Server(
            self._spec, job_name=self.name,
            task_index=self.task, start=False)

    def start(self):
        self._server = self.make_server()
        self._server.start()
        self.connect()

    def get_device(self, name, task=0):
        assert name in self._cluster_dict
        assert task < len(self._cluster_dict[name])

        return f"/job:{name}/task:{task}"

    def count(self, name):
        assert name in self._cluster_dict
        return len(self._cluster_dict[name])
