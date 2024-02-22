import subprocess
from collections.abc import Mapping
from datetime import datetime
from time import time
from uuid import uuid4

import pytz
from boto3 import Session
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session


class RefreshableBotoSession:
    """Boto helper class which lets us create a refreshable session, so that we can cache the client or resource.

    From https://stackoverflow.com/a/69226170/1165181

    Usage
    -----

    ```python
    session = RefreshableBotoSession().refreshable_session()
    client = session.client("s3")  # We now can cache this client object without worrying about expiring credentials.
    ```
    """

    def __init__(self, region_name: str | None = None, profile_name: str | None = None,
                 sts_arn: str | None = None, session_name: str | None = None,
                 session_ttl: int | None = 3000) -> None:
        """Initialize `RefreshableBotoSession`.

        Parameters
        ----------
        region_name: Default region when creating a new connection.

        profile_name: The name of a profile to use.

        sts_arn: The role arn to sts before creating session.

        session_name: An identifier for the assumed role session. It's required when `sts_arn` is given.

        session_ttl: An integer number to set the TTL for each session. Beyond this session, it will renew the token.
            It's fifty minutes by default, which occurs before the default role expiration of 1 hour.
        """

        self.region_name = region_name
        self.profile_name = profile_name
        self.sts_arn = sts_arn
        self.session_name = session_name or uuid4().hex
        self.session_ttl = session_ttl

    def _get_session_credentials(self) -> Mapping[str, str]:
        """Get session credentials."""
        session = Session(region_name=self.region_name, profile_name=self.profile_name)

        if self.sts_arn:  # If `sts_arn` is given, get the credentials by assuming the given role.
            sts_client = session.client(service_name="sts", region_name=self.region_name)
            response = sts_client.assume_role(RoleArn=self.sts_arn, RoleSessionName=self.session_name,
                                              DurationSeconds=self.session_ttl).get("Credentials")

            credentials = {
                "access_key": response.get("AccessKeyId"),
                "secret_key": response.get("SecretAccessKey"),
                "token": response.get("SessionToken"),
                "expiry_time": response.get("Expiration").isoformat(),
            }
        else:
            session_credentials = session.get_credentials()

            if session_credentials:
                frozen_credentials = session_credentials.get_frozen_credentials()
                credentials = {
                    "access_key": frozen_credentials.access_key,
                    "secret_key": frozen_credentials.secret_key,
                    "token": frozen_credentials.token,
                    "expiry_time": datetime.fromtimestamp(
                        time() + self.session_ttl).replace(tzinfo=pytz.utc).isoformat(),
                }
            else:
                credentials = {}

        return credentials

    def refreshable_session(self) -> Session:
        """Get a refreshable boto3 session."""
        session = get_session()
        session._credentials = RefreshableCredentials.create_from_metadata(metadata=self._get_session_credentials(),
                                                                           refresh_using=self._get_session_credentials,
                                                                           method="sts-assume-role")
        session.set_config_variable("region", self.region_name)
        return Session(botocore_session=session)


def s3_sync(source: str, dest: str) -> None:
    subprocess.run(["aws", "s3", "sync", source, dest], check=True)
