import os

subscription_id = os.getenv("SUBSCRIPTION_ID", default="")
resource_group = os.getenv("RESOURCE_GROUP", default="")
workspace_name = os.getenv("WORKSPACE_NAME", default="")
workspace_region = os.getenv("WORKSPACE_REGION", default="")

from azureml.core import Workspace

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()
    print("Workspace configuration succeeded. Skip the workspace creation steps below")
except:
    print("Workspace not accessible. Change your parameters or create a new workspace below")

    from azureml.core import Workspace

    # Create the workspace using the specified parameters
    ws = Workspace.create(name=workspace_name,
                          subscription_id=subscription_id,
                          resource_group=resource_group,
                          location=workspace_region,
                          create_resource_group=True,
                          exist_ok=True)
    ws.get_details()

    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()