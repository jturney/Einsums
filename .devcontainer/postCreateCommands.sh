# Post Create Commands to run after the devcontainer is successfully built

# Activates the einsums environment by adding it to the .zshrc file
echo 'micromamba activate einsums # added by devcontainer.json' >> ~/.zshrc

# Make symbolic links to cmake and ninja to allow non-root user install
sudo ln -s /opt/conda/envs/einsums/bin/cmake /usr/bin/cmake
sudo ln -s /opt/conda/envs/einsums/bin/ninja /usr/bin/ninja

# Copies VSCode helper files to main workspace
if [ -d "/workspaces/Einsums/.vscode" ]; then
  echo "VSCode files already set! Reference files located in .devcontainer/vscode"
else
  cp -r /workspaces/Einsums/.devcontainer/vscode /workspaces/Einsums/.vscode
fi

# Copies  file to main workspace
if [ -f "/workspaces/Einsums/CMakePresets.json" ]; then
  echo "CMakePresets file already set! Reference file located in .devcontainer/CMakePresets.json"
else
  cp -r /workspaces/Einsums/.devcontainer/CMakePresets_ref.json /workspaces/Einsums/CMakePresets.json
fi
