const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  target: "web",
  mode: "production",
  entry: {}, // No JS entry point needed for this simple case
  output: {
    path: path.resolve(__dirname, 'dist', 'truth_chain_frontend'),
    filename: 'index.js',
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: path.join(__dirname, 'src', 'truth_chain_frontend', 'src', 'index.html'),
      filename: 'index.html',
      inject: false, // We are not injecting any JS bundles
    }),
  ],
};
