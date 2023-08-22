"""Create a MAMplan used by mampok."""

from datetime import date, datetime
import yaml
from Bio.Entrez import efetch
from urllib.error import HTTPError

VALID_TOOLS = ["cellxgene-new", "cellxgene-fix", "cellxgene-vip-latest"]
VALID_DATATYPES = ["scrna", "scatac"]
VALID_CLUSTER = ["BN", "GWDGmangt", "GWDG"]


class Mamplan():
    """Class for mamplan creation."""

    def __init__(self, exp_id, files, tool, analyst, datatype,
                 cluster="BN",
                 label=None,
                 mampok_url=True,
                 icinga=False,
                 replicas=1,
                 autoscaling=True,
                 random=False,
                 init_container="s3download",
                 supress_s3=False,
                 organization=["Ag-nerds"],
                 owner=None,
                 user=None,
                 pubmedid=None,
                 citation=None,
                 cpu_limit=None,
                 mem_limit=None,
                 cpu_request=None,
                 mem_request=None):
        """Initialize mamplan object."""

        self.exp_id = exp_id
        self.files = files
        self.tool = tool
        self.analyst = analyst
        self.datatype = datatype
        self.label = label if label else f"{exp_id}: BCU"
        self.bucket = f"mampok-local-bioi-{exp_id}-{tool}"
        self.mampok_url = mampok_url
        self.cluster = cluster
        self.active = False
        self.icinga = icinga
        self.replicas = replicas
        self.autoscaling = autoscaling
        self.random = random
        self.init_container = init_container
        self.supress_s3 = supress_s3
        self.creationdate = date.today().strftime("%d/%m/%Y")
        self.organization = organization
        self.owner = owner
        self.user = user
        self.pubmedid = pubmedid
        self.citation = citation
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit
        self.cpu_request = cpu_request
        self.mem_request = mem_request

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
                "id": self.exp_id,
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

        # TODO Add cpu request and limit

        return mamplan_dict

    def save(self, out):
        """Save object as YAML file."""
        with open(out, 'w') as outfile:
            yaml.dump(self.to_dict(), outfile, default_flow_style=False, default_style=None)

    #######################################################
    #                   Getter & Setter
    #######################################################

    @property
    def exp_id(self):
        return self._exp_id

    @exp_id.setter
    def exp_id(self, exp_id):
        self._exp_id = exp_id.replace("_", "-").lower()

    @property
    def files(self):
        return self._files

    @files.setter
    def files(self, files):
        if not isinstance(files, list):
            files = [files]
        self._files = files

    @property
    def tool(self):
        return self._tool

    @tool.setter
    def tool(self, tool):
        if tool not in VALID_TOOLS:
            raise ValueError(f"Invalid tool given. Valid tools are {VALID_TOOLS}")
        self._tool = tool

    @property
    def analyst(self):
        return self._analyst

    @analyst.setter
    def analyst(self, analyst):
        if not isinstance(analyst, list):
            analyst = [analyst]
        self._analyst = analyst

    @property
    def datatype(self):
        return self._datatype

    @datatype.setter
    def datatype(self, datatype):
        datatype = datatype.lower()
        if datatype not in VALID_DATATYPES:
            raise ValueError(f"Invalid datatype given. Valid datatypes are {VALID_DATATYPES}")
        self._datatype = datatype

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label.replace("_", "-")

    @property
    def bucket(self):
        return self._bucket

    @bucket.setter
    def bucket(self, bucket):
        # TODO bucket name has a char limit check for that limit.
        self._bucket = bucket.replace("_", "-").lower()

    @property
    def mampok_url(self):
        return self._mampok_url

    @mampok_url.setter
    def mampok_url(self, mampok_url):
        if not isinstance(mampok_url, bool):
            raise ValueError("'mampok_url' needs to be a a boolean.")
        self._mampok_url = mampok_url

    @property
    def cluster(self):
        return self._cluster

    @cluster.setter
    def cluster(self, cluster):
        if cluster not in VALID_CLUSTER:
            raise ValueError(f"Invalid cluster given. Valid clusters are {VALID_CLUSTER}")
        self._cluster = cluster

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, active):
        if not isinstance(active, bool):
            raise ValueError("'active' needs to be a boolean.")
        self._active = active

    @property
    def icinga(self):
        return self._icinga

    @icinga.setter
    def icinga(self, icinga):
        if not isinstance(icinga, bool):
            raise ValueError("'icinga' needs to be a boolean.")
        self._icinga = icinga

    @property
    def replicas(self):
        return self._replicas

    @replicas.setter
    def replicas(self, replicas):
        if not isinstance(replicas, int) and replicas < 1:
            raise ValueError("'replicas' needs to be a positiv integer.")
        self._replicas = replicas

    @property
    def autoscaling(self):
        return self._autoscaling

    @autoscaling.setter
    def autoscaling(self, autoscaling):
        if not isinstance(autoscaling, bool):
            raise ValueError("'autoscaling' needs to be a boolean.")
        self._autoscaling = autoscaling

    @property
    def random(self):
        return self._random

    @random.setter
    def random(self, random):
        if not isinstance(random, bool):
            raise ValueError("'random' needs to be a boolean.")
        self._random = random

    @property
    def init_container(self):
        return self._init_container

    @init_container.setter
    def init_container(self, init_container):
        self._init_container = init_container

    @property
    def supress_s3(self):
        return self._supress_s3

    @supress_s3.setter
    def supress_s3(self, supress_s3):
        if not isinstance(supress_s3, bool):
            raise ValueError("'supress_s3' needs to be a boolean.")
        self._supress_s3 = supress_s3

    @property
    def creationdate(self):
        return self._creationdate

    @creationdate.setter
    def creationdate(self, creationdate):
        try:
            datetime.strptime(creationdate, "%d/%m/%Y")
        except ValueError:
            raise ValueError("Incorrect data format, should be DD/MM/YYYY")
        self._creationdate = creationdate

    @property
    def organization(self):
        return self._organization

    @organization.setter
    def organization(self, organization):
        if not isinstance(organization, list):
            organization = [organization]
        # TODO check if organization is valid
        self._organization = organization

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, owner):
        self._owner = owner

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, user):
        if not isinstance(user, list):
            user = [user]
        self._user = user

    @property
    def pubmedid(self):
        return self._pubmedid

    @pubmedid.setter
    def pubmedid(self, pubmedid):
        # Is this check needed? Nice to have but what if pubmed is down or user is offline?
        if pubmedid:
            try:
                efetch(db='pubmed', id=pubmedid, retmode='xml')
            except HTTPError:
                raise ValueError("Invalid pubmed id given.")
        self._pubmedid = pubmedid

    @property
    def citation(self):
        return self._citation

    @citation.setter
    def citation(self, citation):
        if citation and not isinstance(citation, str):
            raise ValueError("'citation' needs to be None or of type str.")
        self._citation = citation
