"use strict";
const loc_aPosition = 1; // constant layout variables.
const loc_aNormal = 2;
const loc_aTexture = 3;
const VSHADER_SOURCE =
`#version 300 es
// Position
layout(location=${loc_aPosition}) in vec4 aPosition;
// Normal
layout(location=${loc_aNormal}) in vec4 aNormal;
// Texture coordinates
layout(location=${loc_aTexture}) in vec2 aTexCoord;


uniform mat4 uMvpMatrix; // MVP Matrix
uniform mat4 uModelMatrix;    // Model matrix
uniform mat4 uNormalMatrix;   // Transformation matrix of the normal

uniform sampler2D earth_disp; // sampler for the earth
uniform sampler2D moon_disp; // sampler for the moon


// bool variable for if statement.
uniform bool uEarth; 
uniform bool uMoon;

uniform float earthHeight; // the earth heightness
uniform float moonHeight; // the moon heightness

out vec2 vTexCoord; // output to fragment shader for texture coordinates
out vec3 vNormal; // output to fragment shader for the specular
out vec3 vPosition; // output local to world position to fragment shader


void main() 
{
  vec4 disp;
  vec4 displace = aPosition;
  
  if(uEarth)
  {
    disp = texture(earth_disp, aTexCoord); //Extracting the color information from the image
    displace.xyz += (earthHeight * disp.rgb) * aNormal.xyz; // Move the actual vertices along the normal axes by earthHeight variable
  }
  else if(uMoon)
  {
    disp = texture(moon_disp, aTexCoord); //Extracting the color information from the image
    displace.xyz += (moonHeight * disp.rgb) * aNormal.xyz; // Move the actual vertices along the normal axes by moonHeight variable
  }
  
  
  gl_Position = uMvpMatrix * displace; // Move the vertices for displace amount

  
  // Calculate the vertex position in the world coordinate
  vPosition = vec3(uModelMatrix * aPosition);
  
  // Calcualte the noraml after transformation
  vNormal = normalize(vec3(uNormalMatrix * aNormal));
  // passing texture coordinates
  vTexCoord = aTexCoord;
  
}`;

// Fragment shader program
const FSHADER_SOURCE = `#version 300 es
precision mediump float; // setting precision mode

uniform vec3 uLightColor;     // Light color
uniform vec3 uLightPosition;  // Position of the light source
uniform vec3 uAmbientLight;   // Ambient light color

uniform sampler2D sun_color; // sun diffuse color
uniform sampler2D earth_color; // earth diffuse color
uniform sampler2D moon_color; // moon diffuse color

uniform sampler2D earth_bump; // earth bumpness amount
uniform sampler2D moon_bump; // moon bumpness amount

uniform sampler2D specularMap; // specular map for the earth

uniform float uMaxDot; // for controlling dot result.

in vec3 vNormal; // normal from the vertex shader after transformation
in vec3 vPosition; // position from the vertex shader
in vec2 vTexCoord; // uv coordinates from the vertex shader
out vec4 fColor; // output color

// bool variables for the if statement
uniform bool uIsSun;
uniform bool uIsEarth;
uniform bool uIsMoon;


// Bump Mapping Unparametrized Surfaces on the GPU by Morten S. Mikkelsen

vec2 dHdxy_fwd(sampler2D bumpMap, vec2 UV, float bumpScale)
{
    // Partial derivative UV by x and y.
    vec2 dSTdx	= dFdx( UV ); 
		vec2 dSTdy	= dFdy( UV );
    // Calculate offset
		float Hll	= bumpScale * texture( bumpMap, UV ).x;

    // Height map is not continuous so it is not differentiable so get the nearing two points by subtracting for the "Central Difference Approximation" method.
		float dBx	= bumpScale * texture( bumpMap, UV + dSTdx ).x - Hll; 
		float dBy	= bumpScale * texture( bumpMap, UV + dSTdy ).x - Hll;
    // return x, y
		return vec2( dBx, dBy );
}

// Pertub the normal to look natural
vec3 pertubNormalArb(vec3 surf_pos, vec3 surf_norm, vec2 dHdxy)
{
    vec3 vSigmaX = vec3( dFdx( surf_pos.x ), dFdx( surf_pos.y ), dFdx( surf_pos.z ) );
		vec3 vSigmaY = vec3( dFdy( surf_pos.x ), dFdy( surf_pos.y ), dFdy( surf_pos.z ) );
		vec3 vN = surf_norm;		// normalized

    // x cross y will yield a normal
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );


		float fDet = dot( vSigmaX, R1 );
		fDet *= ( float( gl_FrontFacing ) * 2.0 - 1.0 );
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
}

void main() 
{
    vec2 dHdxy; // bumpness amount by height map
    vec3 bumpNormal;
    // if is the sun
    if(uIsSun)
      fColor = texture(sun_color, vTexCoord);
    else if(uIsEarth) // if is the earth
    {
      fColor = texture(earth_color, vTexCoord);
      dHdxy = dHdxy_fwd(earth_bump, vTexCoord, 1.0);
    }
    else if(uIsMoon) // if is the moon
    {
      fColor = texture(moon_color, vTexCoord);
      dHdxy = dHdxy_fwd(moon_bump, vTexCoord, 1.0);
    }



    // Normalize the normal because it is interpolated and not 1.0 in length any more
    vec3 normal = normalize(vNormal);

    
    // Calculate the light direction and make its length 1.
    vec3 lightDirection = normalize(uLightPosition - vPosition);



    // The dot product of the light direction and the orientation of a surface (the normal)
    float nDotL;
  
    // use uMaxDot to controll nDotL
    nDotL = max(dot(lightDirection, normal), uMaxDot);



    // Calculate the final color from diffuse reflection and ambient reflection
    vec3 diffuse = uLightColor * fColor.rgb * nDotL;
    vec3 ambient = uAmbientLight * fColor.rgb;
    vec4 specularFactor = texture(specularMap, vTexCoord); //Extracting the color information from the image

    
    
    
    vec3 diffuseBump;
    // in case of the earth or the moon, calculate the bumpness
    if(uIsEarth || uIsMoon)
    {
      // get the bump normal by pertub normal
      bumpNormal = pertubNormalArb(vPosition, normal, dHdxy);
      // give a charateristic color at the bumpness point
      diffuseBump = min(diffuse + dot(bumpNormal, lightDirection), 1.1);
    }

    // Specluar
    vec3 specular = vec3(0.0);
    float shiness = 20.0;
    vec3 lightSpecular = vec3(1.0);

    // earth and nDotL is greater than 0.95
    if(uIsEarth && nDotL > 0.95)
    {
      vec3 v = normalize(-vPosition); // EyePosition
      vec3 r = reflect(-lightDirection, normal); // Reflect from the surface
      specular = lightSpecular * specularFactor.rgb * pow(dot(r, v), shiness);
    }
    
    //Update Final Color
    if(uIsEarth)
      fColor = vec4( (diffuse * diffuseBump + specular) + ambient, fColor.a); // Specular
    else if(uIsMoon)
      fColor = vec4( (diffuse * diffuseBump) + ambient, fColor.a);
    else if(uIsSun)
      fColor = vec4(diffuse + ambient, fColor.a);
}`;

// Default variables for the texture slots.
const SUN = 0;
const EARTH = 5;
const MOON = 10;

// Time variable for the spin animation.
let g_last = Date.now();

// Speed variables for the animation
let EARTH_REVOLVING_SPEED = 60;
let MOON_REVOLVING_SPEED = 30;
let CURRENT_EATHR_REVOLVING_ANGLE = 0;
let CURRENT_MOON_REVOLVING_ANGLE = 0;

let EARTH_ANGLE_SPEED = 60;
let MOON_ANGLE_SPEED = 30;
let CURRENT_EARTH_ANGLE = 0;
let CURRENT_MOON_ANGLE = 0;

let CURRENT_EARTH_HEIGHT = 0;
let CURRENT_MOON_HEIGHT = 0;


function main() 
{
  // Retrieve <canvas> element
  const canvas = document.getElementById('webgl');
  let w = canvas.width;
  let h = canvas.height;
  // Get the rendering context for WebGL
  const gl = canvas.getContext('webgl2');
  if (!gl) {
    console.log('Failed to get the rendering context for WebGL');
    return;
  }

  // Initialize shaders
  if (!initShaders(gl, VSHADER_SOURCE, FSHADER_SOURCE)) {
    console.log('Failed to intialize shaders.');
    return;
  }

  // Turn on depth buffer and culling
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);

  // Create objects and return the vaos
  const sun = create_mesh_sphere(gl, 180, loc_aPosition, loc_aNormal, loc_aTexture, SUN);
  const earth = create_mesh_sphere(gl, 180, loc_aPosition, loc_aNormal, loc_aTexture, EARTH);
  const moon = create_mesh_sphere(gl, 180, loc_aPosition, loc_aNormal, loc_aTexture, MOON);
  
  // Clear background color
  gl.clearColor(0.2, 0.2, 0.2, 1.0);

  // Get the storage locations of uniform variables
  const loc_uMvpMatrix = gl.getUniformLocation(gl.program, 'uMvpMatrix');
  const loc_uModelMatrix = gl.getUniformLocation(gl.program, 'uModelMatrix');
  const loc_uNormalMatrix = gl.getUniformLocation(gl.program, 'uNormalMatrix');

  const loc_uLightPos = gl.getUniformLocation(gl.program, 'uLightPosition');
  const loc_uLightColor = gl.getUniformLocation(gl.program, 'uLightColor');
  const loc_uAmbientColor = gl.getUniformLocation(gl.program, 'uAmbientColor');

  const loc_uSunTrigger = gl.getUniformLocation(gl.program, 'uIsSun');
  const loc_uEarthTrigger = gl.getUniformLocation(gl.program, 'uIsEarth');
  const loc_uMoonTrigger = gl.getUniformLocation(gl.program, 'uIsMoon');

  const loc_uEarthHeight = gl.getUniformLocation(gl.program, 'earthHeight');
  const loc_uMoonHeight = gl.getUniformLocation(gl.program, 'moonHeight');


  // Set uniform variables
  gl.uniform3f(loc_uLightPos, 0, 0, 0);
  // Set the light color
  gl.uniform3f(loc_uLightColor, 1, 1, 1);
  // Set the ambient color
  gl.uniform3f(loc_uAmbientColor, 0.3, 0.3, 0.3);
  // Set the height of the earth and the moon 0
  gl.uniform1f(loc_uEarthHeight, 0);
  gl.uniform1f(loc_uMoonHeight, 0);
 

  // Create Sun, Earth, Moon model and normal matrix
  let SunModelMatrix = new Matrix4();  // Model matrix
  let SunNormalMat = new Matrix4(); // Transformation matrix for normals
  
  let EarthModelMatrix = new Matrix4();
  let EarthNormalMat = new Matrix4();

  let MoonModelMatrix = new Matrix4();
  let MoonNormalMat = new Matrix4();
  
  let MvpMatrix = new Matrix4();    // Model view projection matrix
  
  // Clear color and depth buffer
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);



  // Get dom element by id for slider bars
  document.getElementById("EarthRevolving").oninput = function(ev)
  {
    updateSpeed(0, "EarthRevolving");
  }

  document.getElementById("EarthSpin").oninput = function(ev)
  {
    updateSpeed(1, "EarthSpin");
  }
  
  document.getElementById("EarthHeight").oninput = function(ev)
  {
    updateHeight(gl, 1, "EarthHeight", loc_uEarthHeight);
  }
  
  document.getElementById("MoonRevolving").oninput = function(ev)
  {
    updateSpeed(2, "MoonRevolving");
  }

  document.getElementById("MoonSpin").oninput = function(ev)
  {
    updateSpeed(3, "MoonSpin");
  }

  document.getElementById("MoonHeight").oninput = function(ev)
  {
    updateHeight(gl, 2, "MoonHeight", loc_uMoonHeight);
  }

  

  // Animation function
  let tick = function()
  {
    RotatePerSec();
    RenderPerSec(gl, sun, SunModelMatrix, SunNormalMat, 
      earth, EarthModelMatrix, EarthNormalMat,
      moon, MoonModelMatrix, MoonNormalMat, 
      MvpMatrix, loc_uModelMatrix, loc_uNormalMatrix, loc_uMvpMatrix, 
      loc_uSunTrigger, loc_uEarthTrigger, loc_uMoonTrigger,
      w, h);
    requestAnimationFrame(tick, canvas);
  }
  tick();
}

// Create a sphere
function create_mesh_sphere(gl, SPHERE_DIV, loc_aPosition=0, loc_aNormal=1, loc_aTexCoord=2, type) 
{
  // VAO first
  let vao = gl.createVertexArray();
  // bind VAO
  gl.bindVertexArray(vao);

  let i;
  let j;
  let phi, sin_phi, cos_phi;
  let theta, sin_theta, cos_theta;
  let u, v;
  let p1, p2;

  let positions = [];
  let texcoords = [];
  let indices = [];

  // Generate coordinates
  for (j = 0; j <= SPHERE_DIV; j++) {
    v = 1.0 - j / SPHERE_DIV;
    // Phi for latitude
    phi = (1.0 - v) * Math.PI;
    sin_phi = Math.sin(phi);
    cos_phi = Math.cos(phi);
    for (i = 0; i <= SPHERE_DIV; i++) {
      u = i / SPHERE_DIV;
      // theta for longitude
      theta = u * 2 * Math.PI;
      sin_theta = Math.sin(theta);
      cos_theta = Math.cos(theta);

      positions.push(cos_theta * sin_phi); // x
      positions.push(sin_theta * sin_phi); // y
      positions.push(cos_phi); // z

      texcoords.push(u);
      texcoords.push(v);
    }
  }

  // Generate indices
  // http://www.songho.ca/opengl/gl_sphere.html#sphere

  // p1 -- p1+1
  // |   /  |
  // |  /   |
  // p2 -- p2+1
  for (j = 0; j < SPHERE_DIV; j++) {
    for (i = 0; i < SPHERE_DIV; i++) {
      p1 = j * (SPHERE_DIV + 1) + i;
      p2 = p1 + (SPHERE_DIV + 1);

      indices.push(p1);
      indices.push(p2);
      indices.push(p1 + 1);

      indices.push(p1 + 1);
      indices.push(p2);
      indices.push(p2 + 1);
    }
  }

  // VBO for position
  let buf_position = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf_position);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  gl.vertexAttribPointer(loc_aPosition, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(loc_aPosition);

  // VBO for texture coordinates
  let buf_texcoord = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf_texcoord);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texcoords), gl.STATIC_DRAW);

  // there are two points in the array.
  gl.vertexAttribPointer(loc_aTexCoord, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(loc_aTexCoord);

  ///////////////////////////////////////////////////

  // Create a texture
  let colorTexture = gl.createTexture();

  // Activate texture slots based on the planet types.
  switch (type) {
    case SUN:
      gl.activeTexture(gl.TEXTURE0 + SUN);
      break;
    case EARTH:
      gl.activeTexture(gl.TEXTURE0 + EARTH);
      break;
    case MOON:
      gl.activeTexture(gl.TEXTURE0 + MOON);
      break;
  }

  // Bind texture
  gl.bindTexture(gl.TEXTURE_2D, colorTexture);

  // Set uniform variables for if statement in the shader program.
  // Set sampler ID
  switch (type) {
    case SUN:
      gl.uniform1i(gl.getUniformLocation(gl.program, "sun_color"), SUN);
      break;
    case EARTH:
      gl.uniform1i(gl.getUniformLocation(gl.program, "earth_color"), EARTH);
      break;
    case MOON:
      gl.uniform1i(gl.getUniformLocation(gl.program, "moon_color"), MOON);
      break;
  }

  // Set texture parameters
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array([0, 0, 255, 255])
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

  // load the texture
  let colorMap = new Image();
  let colorUrl;

  // load the texture by type
  // when the load has done bind call loadTexture function to bind texture
  switch (type) {
    case SUN:
      colorUrl =
        "https://live.staticflickr.com/65535/49180228116_c69cd66c06_b.jpg";
      colorMap.src = colorUrl;
      colorMap.onload = function () {
        loadTexture(gl, colorTexture, colorMap);
      };
      requestCORSIfNotSameOrigin(colorMap, colorUrl);
      break;
    case EARTH:
      colorUrl =
        "https://live.staticflickr.com/65535/49180434782_0586a0e9df_b.jpg";
      colorMap.src = colorUrl;
      colorMap.onload = function () {
        loadTexture(gl, colorTexture, colorMap);
      };
      requestCORSIfNotSameOrigin(colorMap, colorUrl);
      break;
    case MOON:
      colorUrl =
        "https://live.staticflickr.com/65535/49179739063_0261f37103_b.jpg";
      colorMap.src = colorUrl;
      colorMap.onload = function () {
        loadTexture(gl, colorTexture, colorMap);
      };
      requestCORSIfNotSameOrigin(colorMap, colorUrl);
      break;
  }

  /////////////////////////////////////////////////////////////////////

  // displacement texture
  let dispTexture = gl.createTexture();

  // increase by 1 to use multiple texutre
  switch (type) {
    case SUN:
      gl.activeTexture(gl.TEXTURE0 + SUN + 1);
      break;
    case EARTH:
      gl.activeTexture(gl.TEXTURE0 + EARTH + 1);
      break;
    case MOON:
      gl.activeTexture(gl.TEXTURE0 + MOON + 1);
      break;
  }

  // bind displacement texutre
  gl.bindTexture(gl.TEXTURE_2D, dispTexture);

  // only earth and the moon have displacement map
  // and assign unique sampler ID
  switch (type) {
    case EARTH:
      gl.uniform1i(gl.getUniformLocation(gl.program, "earth_disp"), EARTH + 1);
      break;
    case MOON:
      gl.uniform1i(gl.getUniformLocation(gl.program, "moon_disp"), MOON + 1);
      break;
  }

  // Set texture parameters
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array([0, 0, 255, 255])
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

  let dispMap = new Image();
  let dispUrl;

  // Same as diffuse color map
  switch (type) {
    case EARTH:
      dispUrl =
        "https://live.staticflickr.com/65535/49180228416_48576644eb_b.jpg";
      dispMap.src = dispUrl;
      dispMap.onload = function () {
        loadTexture(gl, dispTexture, dispMap);
      };
      requestCORSIfNotSameOrigin(dispMap, dispUrl);
      break;
    case MOON:
      dispUrl =
        "https://live.staticflickr.com/65535/49179738753_01c4f2f54c_b.jpg";
      dispMap.src = dispUrl;
      dispMap.onload = function () {
        loadTexture(gl, dispTexture, dispMap);
      };
      requestCORSIfNotSameOrigin(dispMap, dispUrl);
      break;
  }

  ////////////////////////////////////////////////

  let bumpTexture = gl.createTexture();

  // activate texture based on the unique number
  switch (type) {
    case SUN:
      gl.activeTexture(gl.TEXTURE0 + SUN + 2);
      break;
    case EARTH:
      gl.activeTexture(gl.TEXTURE0 + EARTH + 2);
      break;
    case MOON:
      gl.activeTexture(gl.TEXTURE0 + MOON + 2);
      break;
  }

  // Bind texture
  gl.bindTexture(gl.TEXTURE_2D, bumpTexture);

  // Get uniform sampler and assign unique id
  switch (type) {
    case EARTH:
      gl.uniform1i(gl.getUniformLocation(gl.program, "earth_bump"), EARTH + 2);
      break;
    case MOON:
      gl.uniform1i(gl.getUniformLocation(gl.program, "moon_bump"), MOON + 2);
      break;
  }

  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array([0, 0, 255, 255])
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

  let bumpMap = new Image();
  let bumpUrl;

  // load the bumpmap
  switch (type) {
    case EARTH:
      bumpUrl =
        "https://live.staticflickr.com/65535/49180228416_48576644eb_b.jpg";
      bumpMap.src = bumpUrl;
      bumpMap.onload = function () {
        loadTexture(gl, bumpTexture, bumpMap);
      };
      requestCORSIfNotSameOrigin(bumpMap, bumpUrl);
      break;
    case MOON:
      bumpUrl =
        "https://live.staticflickr.com/65535/49179738753_01c4f2f54c_b.jpg";
      bumpMap.src = bumpUrl;
      bumpMap.onload = function () {
        loadTexture(gl, bumpTexture, bumpMap);
      };
      requestCORSIfNotSameOrigin(bumpMap, bumpUrl);
      break;
  }

  //////////////////////////////////////////

  let specularTexture = gl.createTexture();

  switch (type) {
    case SUN:
      gl.activeTexture(gl.TEXTURE0 + SUN + 3);
      break;
    case EARTH:
      gl.activeTexture(gl.TEXTURE0 + EARTH + 3);
      break;
    case MOON:
      gl.activeTexture(gl.TEXTURE0 + MOON + 3);
      break;
  }

  gl.bindTexture(gl.TEXTURE_2D, specularTexture);

  switch (type) {
    case EARTH:
      gl.uniform1i(gl.getUniformLocation(gl.program, "specularMap"), EARTH + 3);
      break;
  }

  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array([0, 0, 255, 255])
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

  let specularMap = new Image();
  let specularUrl;

  switch (type) {
    case EARTH:
      specularUrl =
        "https://live.staticflickr.com/65535/49180228946_6254ab55ab_b.jpg";
      specularMap.src = specularUrl;
      specularMap.onload = function () {
        loadTexture(gl, specularTexture, specularMap);
      };
      requestCORSIfNotSameOrigin(specularMap, specularUrl);
      break;
  }

  /////////////////////////////////////////////////////////

  // Create buffer for normal
  let buf_normal = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf_normal);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  // bind and upload data
  gl.vertexAttribPointer(loc_aNormal, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(loc_aNormal);

  // create a buffer for index buffer
  let buf_index = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buf_index);
  gl.bufferData(
    gl.ELEMENT_ARRAY_BUFFER,
    new Uint16Array(indices),
    gl.STATIC_DRAW
  );

  // Unbind VAO, VBO, EBO
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

  // Return vao and length of indices and type of planet
  return { vao, n: indices.length, type: gl.UNSIGNED_SHORT };
}

// loadTexture will be called after the image loading has done
function loadTexture(gl, texture, image)
{
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
}

// Prevent Cross origin error
function requestCORSIfNotSameOrigin(img, url)
{
  if ((new URL(url)).origin !== window.location.origin) 
  {
    img.crossOrigin = "";
  }
}

// Rotate planets
function RotatePerSec()
{
  let now = Date.now();
  // Calculate the elapsed time
  let elpased = now - g_last;
  g_last = now;

  // Get new angles for the earth and the moon
  CURRENT_MOON_ANGLE = CURRENT_MOON_ANGLE + (MOON_ANGLE_SPEED * elpased) / 1000;
  CURRENT_MOON_REVOLVING_ANGLE = CURRENT_MOON_REVOLVING_ANGLE + (MOON_REVOLVING_SPEED * elpased) / 1000;
  CURRENT_EARTH_ANGLE = CURRENT_EARTH_ANGLE + (EARTH_ANGLE_SPEED * elpased) / 1000;
  CURRENT_EATHR_REVOLVING_ANGLE = CURRENT_EATHR_REVOLVING_ANGLE + (EARTH_REVOLVING_SPEED * elpased) / 1000;

  CURRENT_MOON_ANGLE %= 360;
  CURRENT_MOON_REVOLVING_ANGLE %= 360;
  CURRENT_EARTH_ANGLE %= 360;
  CURRENT_EATHR_REVOLVING_ANGLE %= 360;
}

// Rendering per seconds for animation
function RenderPerSec(gl, sun, SunModelMatrix, SunNormalMat, 
  earth, EarthModelMatrix, EarthNormalMat,
  moon, MoonModelMatrix, MoonNormalMat, 
  MvpMatrix, loc_uModelMatrix, loc_uNormalMatrix, loc_uMvpMatrix, 
  loc_uSunTrigger, loc_uEarthTrigger, loc_uMoonTrigger,
  w, h)
{
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // Get the uniform variables from the vertex shader
  const uEarthFromVSHADER = gl.getUniformLocation(gl.program, 'uEarth');
  const uMoonFromVSHADER = gl.getUniformLocation(gl.program, 'uMoon');
  const loc_uMaxDot = gl.getUniformLocation(gl.program, 'uMaxDot');

  // make a stack for parent - child relation
  let MVPStack = [];
  // Clear MVP matrix
  MvpMatrix.setIdentity();

  // Upper viewport (Sun, earth, moon)
  gl.viewport(0, h/2, w, h/2);

  //Projection
  MvpMatrix.setPerspective(30, w/(h/2), 1, 1000);
  //View
  MvpMatrix.lookAt(0, 10, 15, 0, 0, 0, 0, 1, 0);


  //Model Transforms...

  //Sun

  // Initialize model matrices
  SunModelMatrix.setIdentity();
  EarthModelMatrix.setIdentity();
  MoonModelMatrix.setIdentity();

  // Locate the sun at the 0, 0, 0
  SunModelMatrix.translate(0, 0, 0);
  

  // Accumulate model matrix of the sun
  MvpMatrix.multiply(SunModelMatrix);

  // Push to MVP stack
  MVPStack.push(new Matrix4(MvpMatrix));
  
  // Set unifrom variables for proper rendering
  gl.uniform1i(loc_uSunTrigger, 1);
  gl.uniform1i(loc_uEarthTrigger, 0);
  gl.uniform1i(loc_uMoonTrigger, 0);
  gl.uniform1i(uEarthFromVSHADER, 0);
  gl.uniform1i(uMoonFromVSHADER, 0);
  // Since the sun is the light source so set the uMaxDot to 1.0
  gl.uniform1f(loc_uMaxDot, 1.0);

  //Render Sun
  RenderObj(gl, sun, SunModelMatrix, SunNormalMat, MvpMatrix, loc_uModelMatrix, loc_uNormalMatrix, loc_uMvpMatrix);
  
  //Earth
  // Use model matrix of the sun
  MvpMatrix = MVPStack.pop();

  // Make a camera for the earth 
  let EarthCam = new Matrix4();
  // FOV 60 degree, viewport size w / 2 / h / 2, near plane 1, far plane 1000
  EarthCam.setPerspective(60, (w/2)/(h/2), 1, 1000);
  

  // initialize the model matrix at the origin
  EarthModelMatrix.translate(0, 0, 0);
  // Rotate around the sun by current revolving angle by y axis
  EarthModelMatrix.rotate(CURRENT_EATHR_REVOLVING_ANGLE, 0, 1, 0);
  //Translation from the sun
  EarthModelMatrix.translate(7, 0, 0); 
  
  // Set the camera for the earth at the origin and look at the earth 
  // 0  1  2  3
  // 4  5  6  7
  // 8  9  10 11
  // 12 13 14 15
  // 12, 13, 14 are the x, y, z location for the earth
  // up vector is 0, 1, 0
  EarthCam.lookAt(0, 0, 0, EarthModelMatrix.elements[12], EarthModelMatrix.elements[13], EarthModelMatrix.elements[14], 0, 1, 0);
  // Set the camera for the earth at the center
  EarthCam.translate(0, 0, 0);
  // Rotate the camera by the current earth revolving angle by the y axis
  EarthCam.rotate(CURRENT_EATHR_REVOLVING_ANGLE, 0, 1, 0);
  // Move the camera little bit further to look at the earth
  EarthCam.translate(2.5, 0, 0);

  ///////////////
  // tilt the camera
  EarthCam.rotate(113.5, 1, 0, 0);
  EarthCam.rotate(CURRENT_EARTH_ANGLE, 0, 0, 1);
  /////////////


  // Apply earth model matrix
  MvpMatrix.multiply(EarthModelMatrix);
  
  // push to the stack
  MVPStack.push(new Matrix4(MvpMatrix));

  //////////////
  // earth dynamics apply after pushing the stack
  // Tilt the earth 113.5 degree along the x axis 
  MvpMatrix.rotate(113.5, 1, 0, 0)
  MvpMatrix.rotate(CURRENT_EARTH_ANGLE, 0, 0, 1);
  EarthModelMatrix.rotate(113.5, 1, 0, 0);
  EarthModelMatrix.rotate(CURRENT_EARTH_ANGLE, 0, 0, 1);
  //////////////


  // Set unifrom variables for the earth
  gl.uniform1i(loc_uSunTrigger, 0);
  gl.uniform1i(loc_uEarthTrigger, 1);
  gl.uniform1i(loc_uMoonTrigger, 0);
  gl.uniform1i(uEarthFromVSHADER, 1);
  gl.uniform1i(uMoonFromVSHADER, 0);
  gl.uniform1f(loc_uMaxDot, 0.0);
  
  //Render Earth
  RenderObj(gl, earth, EarthModelMatrix, EarthNormalMat, MvpMatrix, loc_uModelMatrix, loc_uNormalMatrix, loc_uMvpMatrix);


  //Moon, translation and rotation itself
  MvpMatrix = MVPStack.pop();

  // Create a camera for the moon
  let MoonCam = new Matrix4();
  // right bottom
  MoonCam.setPerspective(60, (w/2)/(h/2), 1, 1000);
 

  // makes the moon bit smaller than the earth
  MoonModelMatrix.scale(0.8, 0.8, 0.8);
  // translate at the origin.
  MoonModelMatrix.translate(0, 0, 0);
  // rotate around the earth along the y axis
  MoonModelMatrix.rotate(CURRENT_MOON_REVOLVING_ANGLE, 0, 1, 0);
  // move the moon along the x axis
  MoonModelMatrix.translate(3, 0, 0);
  
  // Set the camera for the moon
  MoonCam.lookAt(0, 0, 0, MoonModelMatrix.elements[12], MoonModelMatrix.elements[13], MoonModelMatrix.elements[14], 0, 1, 0);
  MoonCam.rotate(CURRENT_MOON_REVOLVING_ANGLE, 0, 1, 0);
  MoonCam.translate(2.5, 0, 0);
  MoonCam.rotate(CURRENT_MOON_ANGLE, 0, 1, 0);
  
  
  MvpMatrix.multiply(MoonModelMatrix);

  MoonModelMatrix.rotate(CURRENT_MOON_ANGLE, 0, 1, 0);
  
  MVPStack.push(new Matrix4(MvpMatrix));
  
  // Set uniform variables for the moon
  gl.uniform1i(loc_uSunTrigger, 0);
  gl.uniform1i(loc_uEarthTrigger, 0);
  gl.uniform1i(loc_uMoonTrigger, 1);
  gl.uniform1i(uEarthFromVSHADER, 0);
  gl.uniform1i(uMoonFromVSHADER, 1);
  gl.uniform1f(loc_uMaxDot, 0.0);


  //Render Moon
  RenderObj(gl, moon, MoonModelMatrix, MoonNormalMat, MvpMatrix, loc_uModelMatrix, loc_uNormalMatrix, loc_uMvpMatrix);

  /////Render bottom left viewport

  gl.viewport(0, 0, w/2, h/2);

  gl.uniform1i(loc_uSunTrigger, 0);
  gl.uniform1i(loc_uEarthTrigger, 1);
  gl.uniform1i(loc_uMoonTrigger, 0);
  gl.uniform1i(uEarthFromVSHADER, 1);
  gl.uniform1i(uMoonFromVSHADER, 0);
  gl.uniform1f(loc_uMaxDot, 0.0);

  //Render Earth
  RenderObj(gl, earth, EarthModelMatrix, EarthNormalMat, EarthCam, loc_uModelMatrix, loc_uNormalMatrix, loc_uMvpMatrix);

  ////Render bottom right viewport

  gl.viewport(w/2, 0, w/2, h/2);

  gl.uniform1i(loc_uSunTrigger, 0);
  gl.uniform1i(loc_uEarthTrigger, 0);
  gl.uniform1i(loc_uMoonTrigger, 1);
  gl.uniform1i(uEarthFromVSHADER, 0);
  gl.uniform1i(uMoonFromVSHADER, 1);
  gl.uniform1f(loc_uMaxDot, 0.0);

  //Render Moon
  RenderObj(gl, moon, MoonModelMatrix, MoonNormalMat, MoonCam, loc_uModelMatrix, loc_uNormalMatrix, loc_uMvpMatrix);
}

function RenderObj(gl, obj, ModelMatrix, NormalMatrix, MvpMatrix, loc_uModelMatrix, loc_uNormalMatrix, loc_uMvpMatrix)
{
  // Bind VAO
  gl.bindVertexArray(obj.vao);

  // Model matrix has changed so normal matrix should be calculated again
  //Normal transformation
  NormalMatrix.setInverseOf(ModelMatrix);
  NormalMatrix.transpose();

  // Pass the model matrix to uModelMatrix
  gl.uniformMatrix4fv(loc_uModelMatrix, false, ModelMatrix.elements);

  // Pass the model view projection matrix to umvpMatrix
  gl.uniformMatrix4fv(loc_uMvpMatrix, false, MvpMatrix.elements);

  // Pass the transformation matrix for normals to uNormalMatrix
  gl.uniformMatrix4fv(loc_uNormalMatrix, false, NormalMatrix.elements);

  gl.drawElements(gl.TRIANGLES, obj.n, obj.type, 0);
}

// Get slider bar from dom and update the speed
function updateSpeed(option, id)
{
  switch(option)
  {
    case 0 :
      EARTH_REVOLVING_SPEED = parseInt(document.getElementById(id).value);
      break;
    case 1 :
      EARTH_ANGLE_SPEED = parseInt(document.getElementById(id).value);
      break;
    case 2 :
      MOON_REVOLVING_SPEED = parseInt(document.getElementById(id).value);
      break;
    case 3 :
      MOON_ANGLE_SPEED = parseInt(document.getElementById(id).value);
      break;
  }
}

// Get Height value from the dom and update.
function updateHeight(gl, option, id, loc)
{
  switch(option)
  {
    case 1 :
      CURRENT_EARTH_HEIGHT = parseFloat(document.getElementById(id).value);
      console.log(CURRENT_EARTH_HEIGHT);
      gl.uniform1f(loc, CURRENT_EARTH_HEIGHT);
      break;
    case 2 :
      CURRENT_MOON_HEIGHT = parseFloat(document.getElementById(id).value);
      console.log(CURRENT_MOON_HEIGHT);
      gl.uniform1f(loc, CURRENT_MOON_HEIGHT);
      break;
  }
}