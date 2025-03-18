import bw2data as bd
import bw2io as bi

# set project
ei_version = "3.10"
project_name = "internalizer_ei_{}".format(ei_version)

# import ecoinvent databases
bd.projects.set_current(project_name)
bi.import_ecoinvent_release(
    version=ei_version,
    system_model="cutoff",
    username="pik",
    password="8FragAvBar!"
)