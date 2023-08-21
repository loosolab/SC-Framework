"""Create a MAMplan used by mampok."""

from datetime import date
import yaml

VALID_TOOLS = ["cellxgene-new", "cellxgene-fix", "cellxgene-vip-latest"]
VALID_DATATYPES = ["scrna", "scatac"]
VALID_CLUSTER = ["BN", "GWDGmangt", "GWDG"]


class Mamplan():
    """Class for mamplan creation."""

    def __init__(self, id, files, tool, analyst, datatype):
        """Initialize mamplan object."""

        id = id.replace("_", "-")

        if not isinstance(analyst, list):
            analyst = [analyst]

        if not isinstance(files, list):
            files = [files]

        self.id = id
        self.files = files
        self.tool = tool
        self.analyst = analyst
        self.datatype = datatype
        self.label = f"{id}: BCU"
        self.bucket = f"mampok-local-bioi-{id}-{tool}"
        self.mampok_url = True
        self.mampok_url = True
        self.cluster = "BN"
        self.active = False
        self.icinga = False
        self.replicas = 1
        self.autoscaling = True
        self.random = False
        self.init_container = "s3download"
        self.supress_s3 = False
        self.creationdate = date.today().strftime("%d/%m/%Y")
        self.organization = ["AG-nerds"]
        self.owner = None
        self.user = None
        self.pubmedid = None
        self.citation = None

    def to_dict(self):
        """Return object as dictionary."""

        mamplan_dict = {
            "container": {"main": {"extra_args": ["-t", self.label]}},
            "deployment": {
                "mampok_url": self.mampok_url,
                "cluster": self.cluster,
                "active": self.active,
                "bucket": self.bucket,
                "icinga": self.icinga,
                "replicas": self.replicas,
                "autoscaling": self.autoscaling,
                "random": self.random
            },
            "project": {
                "files": self.files,
                "id": self.id,
                "init_container": self.init_container,
                "supress_s3": self.supress_s3,
                "tool": self.tool,
            },
            "tags": {
                "analyst": self.analyst,
                "creationdate": self.creationdate,
                "datatype": self.datatype,
                "organization": self.organization
            }
        }

        if self.owner:
            mamplan_dict["tags"]["owner"] = self.owner

        if self.user:
            mamplan_dict["tags"]["user"] = self.user

        if self.pubmedid:
            mamplan_dict["tags"]["pubmedid"] = self.pubmedid

        if self.citation:
            mamplan_dict["tags"]["citation"] = self.citation

        return mamplan_dict

    def save(self, out):
        """Save object as YAML file."""
        with open(out, 'w') as outfile:
            yaml.dump(self.to_dict(), outfile, default_flow_style=False, default_style=None)
