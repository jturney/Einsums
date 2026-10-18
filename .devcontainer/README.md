# Using the development container

Container is set to use Ubuntu 24.04 (Noble) by default. Docker image accepts two arguments:
 - `VARIANT` - Accepts the base tags from [buildpack-deps](https://hub.docker.com/_/buildpack-deps/) for an Ubuntu image
 - `INSTALL_INTEL` - Boolean that controls whether or not to install the Intel HPC Toolkit, which includes VTune
Nearly all packages are installed via micromamba to use the latest versions with the list of packages in the einsums_env.yml file.
Optionally, VSCode files for the container are provided and set to the main workspace directory if .vscode/ is not already present.
Same for CMakePresets.json, one is provided if not already present.

# Running VTune inside of Docker container

This uses the
new-ish [VTune web interface](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/launch/web-server-ui.html).

    vtune-backend --enable-server-profiling --no-https

`vtune-backend` will print some information to the screen including something like:

    Serving GUI at https://127.0.0.1:YYYYY?one-time-token=XXXXXXXX

Ctrl+Click the link and VSCode will open a new browser window with the VTune graphical interface. VSCode will forward
port automatically. If it doesn't work try running:

    vtune-backend --no-https --reset-passphrase

Open the given web address and create a new password for the VTune web interface. Once this is done you can
Ctrl+C `vtune-backend` and run the first `vtune-backend` command above.

When running an executable in VTune be sure to use the full path of the executable. For example, when running `test-all`
use:

    /workspaces/EinsumsInCpp/build/tests/test-all
