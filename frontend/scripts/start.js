'use strict';

// react-scripts 5 creates its webpack compiler before webpack-dev-server gets
// a chance to apply HotModuleReplacementPlugin. With newer Node/npm installs,
// that can produce a contradictory bundle: the HMR client is included, but
// `module.hot` is compiled to `false` and the app crashes at startup.
// Wrap CRA's config factory so the plugin is present before compiler creation.
const Module = require('module');
const webpack = require('webpack');
const originalLoad = Module._load;

Module._load = function loadWithHmr(request, parent, isMain) {
  const loaded = originalLoad.call(this, request, parent, isMain);
  const fromCraStart = parent && /react-scripts[/\\]scripts[/\\]start\.js$/.test(parent.filename);

  if (fromCraStart && request === '../config/webpack.config' && typeof loaded === 'function') {
    return webpackEnvironment => {
      const config = loaded(webpackEnvironment);
      if (webpackEnvironment === 'development') {
        const alreadyEnabled = config.plugins.some(
          plugin => plugin && plugin.constructor === webpack.HotModuleReplacementPlugin
        );
        if (!alreadyEnabled) config.plugins.push(new webpack.HotModuleReplacementPlugin());
      }
      return config;
    };
  }
  return loaded;
};

require('react-scripts/scripts/start');
