## TODO's:
- separate soft and hard versions of 
grid space manipulator nodes, but make
  them have identicle node interfaces so
  that nodes trained on small `SoftGridSpaceManipulatorNode`'s
  can easily scale to large `HardGridSpaceManipulatorNode`'s.
  
This is useful for:
- very large long-term memory / working memory