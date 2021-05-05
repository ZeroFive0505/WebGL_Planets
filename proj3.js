"use strict";
const loc_aPosition = 1;
const loc_aNormal = 5;
const loc_aTexture = 7;
const VSHADER_SOURCE =
`#version 300 es
layout(location=${loc_aPosition}) in vec4 aPosition;
layout(location=${loc_aNormal}) in vec4 aNormal;
layout(location=${loc_aTexture}) in vec2 aTexCoord;


uniform mat4 uMvpMatrix;
uniform mat4 uModelMatrix;    // Model matrix
uniform mat4 uNormalMatrix;   // Transformation matrix of the normal

uniform sampler2D earth_disp;
uniform sampler2D moon_disp;


uniform bool uEarth;
uniform bool uMoon;

uniform float earthHeight;
uniform float moonHeight;

out vec2 vTexCoord;
out vec3 vNormal;
out vec3 vPosition;


void main() 
{
  
  vec4 disp;
  vec4 displace = aPosition;
  
  if(uEarth)
  {
    disp = texture(earth_disp, aTexCoord); //Extracting the color information from the image
    displace.xyz += (earthHeight * disp.rgb) * aNormal.xyz;
  }
  else if(uMoon)
  {
    disp = texture(moon_disp, aTexCoord); //Extracting the color information from the image
    displace.xyz += (moonHeight * disp.rgb) * aNormal.xyz;
  }
  
  
  gl_Position = uMvpMatrix * displace;

  
  // Calculate the vertex position in the world coordinate
  vPosition = vec3(uModelMatrix * aPosition);
  
  vNormal = normalize(vec3(uNormalMatrix * aNormal));
  vTexCoord = aTexCoord;
  
}`;

// Fragment shader program
const FSHADER_SOURCE =
`#version 300 es
precision mediump float;

uniform vec3 uLightColor;     // Light color
uniform vec3 uLightPosition;  // Position of the light source
uniform vec3 uAmbientLight;   // Ambient light color

uniform sampler2D sun_color;
uniform sampler2D earth_color;
uniform sampler2D moon_color;

uniform sampler2D earth_bump;
uniform sampler2D moon_bump;

uniform sampler2D specularMap;

uniform float uMaxDot;


in vec3 vNormal;
in vec3 vPosition;
in vec2 vTexCoord;
out vec4 fColor;

uniform bool uIsSun;
uniform bool uIsEarth;
uniform bool uIsMoon;



vec2 dHdxy_fwd(sampler2D bumpMap, vec2 UV, float bumpScale)
{
    vec2 dSTdx	= dFdx( UV );
		vec2 dSTdy	= dFdy( UV );
		float Hll	= bumpScale * texture( bumpMap, UV ).x;
		float dBx	= bumpScale * texture( bumpMap, UV + dSTdx ).x - Hll;
		float dBy	= bumpScale * texture( bumpMap, UV + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
}

vec3 pertubNormalArb(vec3 surf_pos, vec3 surf_norm, vec2 dHdxy)
{
    vec3 vSigmaX = vec3( dFdx( surf_pos.x ), dFdx( surf_pos.y ), dFdx( surf_pos.z ) );
		vec3 vSigmaY = vec3( dFdy( surf_pos.x ), dFdy( surf_pos.y ), dFdy( surf_pos.z ) );
		vec3 vN = surf_norm;		// normalized
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 );
		fDet *= ( float( gl_FrontFacing ) * 2.0 - 1.0 );
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
}



void main() 
{
    vec2 dHdxy;
    vec3 bumpNormal;
    if(uIsSun)
      fColor = texture(sun_color, vTexCoord);
    else if(uIsEarth)
    {
      fColor = texture(earth_color, vTexCoord);
      dHdxy = dHdxy_fwd(earth_bump, vTexCoord, 1.0);
    }
    else if(uIsMoon)
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
  
    nDotL = max(dot(lightDirection, normal), uMaxDot);



    // Calculate the final color from diffuse reflection and ambient reflection
    vec3 diffuse = uLightColor * fColor.rgb * nDotL;
    vec3 ambient = uAmbientLight * fColor.rgb;
    vec4 specularFactor = texture(specularMap, vTexCoord); //Extracting the color information from the image

    
    
    
    vec3 diffuseBump;
    if(uIsEarth || uIsMoon)
    {
      bumpNormal = pertubNormalArb(vPosition, normal, dHdxy);
      diffuseBump = min(diffuse + dot(bumpNormal, lightDirection), 1.1);
    }

    vec3 specular = vec3(0.0);
    float shiness = 20.0;
    vec3 lightSpecular = vec3(1.0);

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


const SUN = 0;
const EARTH = 5;
const MOON = 10;


let g_last = Date.now();

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

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);

  const sun = create_mesh_sphere(gl, 180, loc_aPosition, loc_aNormal, loc_aTexture, SUN);
  const earth = create_mesh_sphere(gl, 180, loc_aPosition, loc_aNormal, loc_aTexture, EARTH);
  const moon = create_mesh_sphere(gl, 180, loc_aPosition, loc_aNormal, loc_aTexture, MOON);
  
  // Set the clear color and enable the depth test
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


  gl.uniform3f(loc_uLightPos, 0, 0, 0);
  gl.uniform3f(loc_uLightColor, 1, 1, 1);
  gl.uniform3f(loc_uAmbientColor, 0.3, 0.3, 0.3);
  gl.uniform1f(loc_uEarthHeight, 0);
  gl.uniform1f(loc_uMoonHeight, 0);
 

  let SunModelMatrix = new Matrix4();  // Model matrix
  let SunNormalMat = new Matrix4(); // Transformation matrix for normals
  
  let EarthModelMatrix = new Matrix4();
  let EarthNormalMat = new Matrix4();

  let MoonModelMatrix = new Matrix4();
  let MoonNormalMat = new Matrix4();
  
  let MvpMatrix = new Matrix4();    // Model view projection matrix
  
  // Clear color and depth buffer
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);



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

  gl.bindVertexArray(null);
}


function create_mesh_sphere(gl, SPHERE_DIV, loc_aPosition=0, loc_aNormal=1, loc_aTexCoord=2, type) 
{ // Create a sphere
    let vao = gl.createVertexArray();
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
    for (j = 0; j <= SPHERE_DIV; j++)
    {
        v = 1.0 - j/SPHERE_DIV;
        phi = (1.0-v) * Math.PI;
        sin_phi = Math.sin(phi);
        cos_phi = Math.cos(phi);
        for (i = 0; i <= SPHERE_DIV; i++)
        {
            u = i/SPHERE_DIV;
            theta = u * 2 * Math.PI;
            sin_theta = Math.sin(theta);
            cos_theta = Math.cos(theta);
            
            positions.push(cos_theta * sin_phi);  // x
            positions.push(sin_theta * sin_phi);  // y
            positions.push(cos_phi);       // z

            texcoords.push(u);
            texcoords.push(v);
        }
    }
    
    // Generate indices
    for (j = 0; j < SPHERE_DIV; j++)
    {
        for (i = 0; i < SPHERE_DIV; i++)
        {
            p1 = j * (SPHERE_DIV+1) + i;
            p2 = p1 + (SPHERE_DIV+1);
            
            indices.push(p1);
            indices.push(p2);
            indices.push(p1 + 1);
            
            indices.push(p1 + 1);
            indices.push(p2);
            indices.push(p2 + 1);
        }
    }
    
    let buf_position = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf_position);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    
    gl.vertexAttribPointer(loc_aPosition, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc_aPosition);

    let buf_texcoord = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf_texcoord);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texcoords), gl.STATIC_DRAW);
 
    gl.vertexAttribPointer(loc_aTexCoord, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc_aTexCoord);



    ///////////////////////////////////////////////////


    let colorTexture = gl.createTexture();
    

    
    switch(type)
    {
      case SUN :
        gl.activeTexture(gl.TEXTURE0 + SUN);
        break;
      case EARTH :
        gl.activeTexture(gl.TEXTURE0 + EARTH);
        break;
      case MOON :
        gl.activeTexture(gl.TEXTURE0 + MOON);
        break;
    }

    gl.bindTexture(gl.TEXTURE_2D, colorTexture);
    
    
    switch(type)
    {
      case SUN :
        gl.uniform1i(gl.getUniformLocation(gl.program, 'sun_color'), SUN);
        break;
      case EARTH :
        gl.uniform1i(gl.getUniformLocation(gl.program, 'earth_color'), EARTH);
        break;
      case MOON :
        gl.uniform1i(gl.getUniformLocation(gl.program, 'moon_color'), MOON);
        break;
    }

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    
    
    let colorMap = new Image();
    let colorUrl;


    switch(type)
    {
      case SUN :
        colorUrl = 'https://live.staticflickr.com/65535/49180228116_c69cd66c06_b.jpg';
        colorMap.src = colorUrl;
        colorMap.onload = function()
        {
          loadTexture(gl, colorTexture, colorMap);
        }
        requestCORSIfNotSameOrigin(colorMap, colorUrl);
        break;
      case EARTH :
        colorUrl = 'https://live.staticflickr.com/65535/49180434782_0586a0e9df_b.jpg';
        colorMap.src = colorUrl;
        colorMap.onload = function()
        {
          loadTexture(gl, colorTexture, colorMap);
        }    
        requestCORSIfNotSameOrigin(colorMap, colorUrl);
        break;
      case MOON :
        colorUrl = 'https://live.staticflickr.com/65535/49179739063_0261f37103_b.jpg';
        colorMap.src = colorUrl;
        colorMap.onload = function()
        {
          loadTexture(gl, colorTexture, colorMap);
        }
        requestCORSIfNotSameOrigin(colorMap, colorUrl);
        break;
    }


    /////////////////////////////////////////////////////////////////////


    let dispTexture = gl.createTexture();

    switch(type)
    {
      case SUN :
        gl.activeTexture(gl.TEXTURE0 + SUN + 1);
        break;
      case EARTH :
        gl.activeTexture(gl.TEXTURE0 + EARTH + 1);
        break;
      case MOON :
        gl.activeTexture(gl.TEXTURE0 + MOON + 1);
        break;
    }

    gl.bindTexture(gl.TEXTURE_2D, dispTexture);
   


    switch(type)
    {
      case EARTH :
        gl.uniform1i(gl.getUniformLocation(gl.program, 'earth_disp'), EARTH + 1);
        break;
      case MOON :
        gl.uniform1i(gl.getUniformLocation(gl.program, 'moon_disp'), MOON + 1);
        break;
    }


    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    

    let dispMap = new Image();
    let dispUrl;


    switch(type)
    {
      case EARTH :
        dispUrl = 'https://live.staticflickr.com/65535/49180228416_48576644eb_b.jpg';
        dispMap.src = dispUrl;
        dispMap.onload = function()
        {
          loadTexture(gl, dispTexture, dispMap);
        }
        requestCORSIfNotSameOrigin(dispMap, dispUrl);
        break;
      case MOON :
        dispUrl = 'https://live.staticflickr.com/65535/49179738753_01c4f2f54c_b.jpg';
        dispMap.src = dispUrl;
        dispMap.onload = function()
        {
          loadTexture(gl, dispTexture, dispMap);
        }
        requestCORSIfNotSameOrigin(dispMap, dispUrl);
        break;
    }


    ////////////////////////////////////////////////



    let bumpTexture = gl.createTexture();


    switch(type)
    {
      case SUN :
        gl.activeTexture(gl.TEXTURE0 + SUN + 2);
        break;
      case EARTH :
        gl.activeTexture(gl.TEXTURE0 + EARTH + 2);
        break;
      case MOON :
        gl.activeTexture(gl.TEXTURE0 + MOON + 2);
        break;
    }

    gl.bindTexture(gl.TEXTURE_2D, bumpTexture);
   


    switch(type)
    {
      case EARTH :
        gl.uniform1i(gl.getUniformLocation(gl.program, 'earth_bump'), EARTH + 2);
        break;
      case MOON :
        gl.uniform1i(gl.getUniformLocation(gl.program, 'moon_bump'), MOON + 2);
        break;
    }


    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    

    let bumpMap = new Image();
    let bumpUrl;


    switch(type)
    {
      case EARTH :
        bumpUrl = 'https://live.staticflickr.com/65535/49180228416_48576644eb_b.jpg';
        bumpMap.src = bumpUrl;
        bumpMap.onload = function()
        {
          loadTexture(gl, bumpTexture, bumpMap);
        }
        requestCORSIfNotSameOrigin(bumpMap, bumpUrl);
        break;
      case MOON :
        bumpUrl = 'https://live.staticflickr.com/65535/49179738753_01c4f2f54c_b.jpg';
        bumpMap.src = bumpUrl;
        bumpMap.onload = function()
        {
          loadTexture(gl, bumpTexture, bumpMap);
        }
        requestCORSIfNotSameOrigin(bumpMap, bumpUrl);
        break;
    }


    //////////////////////////////////////////

    let specularTexture = gl.createTexture();


    switch(type)
    {
      case SUN :
        gl.activeTexture(gl.TEXTURE0 + SUN + 3);
        break;
      case EARTH :
        gl.activeTexture(gl.TEXTURE0 + EARTH + 3);
        break;
      case MOON :
        gl.activeTexture(gl.TEXTURE0 + MOON + 3);
        break;
    }

    gl.bindTexture(gl.TEXTURE_2D, specularTexture);
    


    switch(type)
    {
      case EARTH :
        gl.uniform1i(gl.getUniformLocation(gl.program, 'specularMap'), EARTH + 3);
        break;
    }

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    

    let specularMap = new Image();
    let specularUrl;


    switch(type)
    {
      case EARTH :
        specularUrl = 'https://live.staticflickr.com/65535/49180228946_6254ab55ab_b.jpg';
        specularMap.src = specularUrl;
        specularMap.onload = function()
        {
          loadTexture(gl, specularTexture, specularMap);
        }
        requestCORSIfNotSameOrigin(specularMap, specularUrl);
        break;
    }

    /////////////////////////////////////////////////////////



    let buf_normal = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf_normal);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    
    gl.vertexAttribPointer(loc_aNormal, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc_aNormal);

    let buf_index = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buf_index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    
    return {vao, n : indices.length, type : gl.UNSIGNED_SHORT}; 
}


function loadTexture(gl, texture, image)
{
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
}


function requestCORSIfNotSameOrigin(img, url)
{
  if ((new URL(url)).origin !== window.location.origin) 
  {
    img.crossOrigin = "";
  }
}


function RotatePerSec()
{
  let now = Date.now();
  let elpased = now - g_last;
  g_last = now;


  CURRENT_MOON_ANGLE = CURRENT_MOON_ANGLE + (MOON_ANGLE_SPEED * elpased) / 1000;
  CURRENT_MOON_REVOLVING_ANGLE = CURRENT_MOON_REVOLVING_ANGLE + (MOON_REVOLVING_SPEED * elpased) / 1000;
  CURRENT_EARTH_ANGLE = CURRENT_EARTH_ANGLE + (EARTH_ANGLE_SPEED * elpased) / 1000;
  CURRENT_EATHR_REVOLVING_ANGLE = CURRENT_EATHR_REVOLVING_ANGLE + (EARTH_REVOLVING_SPEED * elpased) / 1000;

  CURRENT_MOON_ANGLE %= 360;
  CURRENT_MOON_REVOLVING_ANGLE %= 360;
  CURRENT_EARTH_ANGLE %= 360;
  CURRENT_EATHR_REVOLVING_ANGLE %= 360;
}

function RenderPerSec(gl, sun, SunModelMatrix, SunNormalMat, 
  earth, EarthModelMatrix, EarthNormalMat,
  moon, MoonModelMatrix, MoonNormalMat, 
  MvpMatrix, loc_uModelMatrix, loc_uNormalMatrix, loc_uMvpMatrix, 
  loc_uSunTrigger, loc_uEarthTrigger, loc_uMoonTrigger,
  w, h)
{
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const uEarthFromVSHADER = gl.getUniformLocation(gl.program, 'uEarth');
  const uMoonFromVSHADER = gl.getUniformLocation(gl.program, 'uMoon');
  const loc_uMaxDot = gl.getUniformLocation(gl.program, 'uMaxDot');


  let MVPStack = [];
  MvpMatrix.setIdentity();

  gl.viewport(0, h/2, w, h/2);

  //Projection
  MvpMatrix.setPerspective(30, w/(h/2), 1, 1000);
  //View
  MvpMatrix.lookAt(0, 10, 15, 0, 0, 0, 0, 1, 0);


  //Model Transforms...

  //Sun

  SunModelMatrix.setIdentity();
  EarthModelMatrix.setIdentity();
  MoonModelMatrix.setIdentity();

  SunModelMatrix.translate(0, 0, 0);
  

  MvpMatrix.multiply(SunModelMatrix);

  MVPStack.push(new Matrix4(MvpMatrix));
  

  gl.uniform1i(loc_uSunTrigger, 1);
  gl.uniform1i(loc_uEarthTrigger, 0);
  gl.uniform1i(loc_uMoonTrigger, 0);
  gl.uniform1i(uEarthFromVSHADER, 0);
  gl.uniform1i(uMoonFromVSHADER, 0);
  gl.uniform1f(loc_uMaxDot, 1.0);

  //Render Sun
  RenderObj(gl, sun, SunModelMatrix, SunNormalMat, MvpMatrix, loc_uModelMatrix, loc_uNormalMatrix, loc_uMvpMatrix);
  
  //Earth
  MvpMatrix = MVPStack.pop();


  let EarthCam = new Matrix4();
  EarthCam.setPerspective(60, (w/2)/(h/2), 1, 1000);
  

  EarthModelMatrix.translate(0, 0, 0);
  EarthModelMatrix.rotate(CURRENT_EATHR_REVOLVING_ANGLE, 0, 1, 0);
  EarthModelMatrix.translate(7, 0, 0); //Translation from the sun
  
  EarthCam.lookAt(0, 0, 0, EarthModelMatrix.elements[12], EarthModelMatrix.elements[13], EarthModelMatrix.elements[14], 0, 1, 0);
  EarthCam.translate(0, 0, 0);
  EarthCam.rotate(CURRENT_EATHR_REVOLVING_ANGLE, 0, 1, 0);
  EarthCam.translate(2.5, 0, 0);

  ///////////////
  EarthCam.rotate(113.5, 1, 0, 0);
  EarthCam.rotate(CURRENT_EARTH_ANGLE, 0, 0, 1);
  /////////////


  MvpMatrix.multiply(EarthModelMatrix);
  
  MVPStack.push(new Matrix4(MvpMatrix));

  //////////////
  MvpMatrix.rotate(113.5, 1, 0, 0)
  MvpMatrix.rotate(CURRENT_EARTH_ANGLE, 0, 0, 1);
  EarthModelMatrix.rotate(113.5, 1, 0, 0);
  EarthModelMatrix.rotate(CURRENT_EARTH_ANGLE, 0, 0, 1);
  //////////////


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

  let MoonCam = new Matrix4();
  MoonCam.setPerspective(60, (w/2)/(h/2), 1, 1000);
 

  MoonModelMatrix.scale(0.8, 0.8, 0.8);
  MoonModelMatrix.translate(0, 0, 0);
  MoonModelMatrix.rotate(CURRENT_MOON_REVOLVING_ANGLE, 0, 1, 0);
  MoonModelMatrix.translate(3, 0, 0);
  
  
  MoonCam.lookAt(0, 0, 0, MoonModelMatrix.elements[12], MoonModelMatrix.elements[13], MoonModelMatrix.elements[14], 0, 1, 0);
  MoonCam.rotate(CURRENT_MOON_REVOLVING_ANGLE, 0, 1, 0);
  MoonCam.translate(2.5, 0, 0);
  MoonCam.rotate(CURRENT_MOON_ANGLE, 0, 1, 0);
  
  
  MvpMatrix.multiply(MoonModelMatrix);
  
  MVPStack.push(new Matrix4(MvpMatrix));
  
  MoonModelMatrix.rotate(CURRENT_MOON_ANGLE, 0, 1, 0);
  
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
  gl.bindVertexArray(obj.vao);


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