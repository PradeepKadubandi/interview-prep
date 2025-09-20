# Copyright Exafunction, Inc.

"""Remote file replicator between a source and target directory."""

import posixpath
import dataclasses
from typing import Any, Callable

from file_system import FileSystem
from file_system import FileSystemEvent
from file_system import FileSystemEventType

# If you're completing this task in an online assessment, you can increment this
# constant to enable more unit tests relevant to the task you are on (1-5).
TASK_NUM = 1

# TODO: Can create a generic request class and put path in there and make others sub-classes of it.
@dataclasses.dataclass
class AddDirectoryRequest:
    path: str

# TODO: Create subclasses of nuanced responses if needed.
@dataclasses.dataclass
class GenericResponse:
    success: bool

@dataclasses.dataclass
class AddFileRequest:
    path: str
    content: str

@dataclasses.dataclass
class RemoveFileRequest:
    path: str

@dataclasses.dataclass
class RemoveDirectoryRequest:
    path: str
    


class ReplicatorSource:
    """Class representing the source side of a file replicator."""

    def __init__(self, fs: FileSystem, dir_path: str, rpc_handle: Callable[[Any], Any]):
        self._fs = fs
        self._dir_path = dir_path
        self._rpc_handle = rpc_handle

        # TODO
        self.replicate_dir_recursively(self._dir_path)

    def replicate_dir_recursively(self, sub_dir_path: str):
        self._fs.watchdir(sub_dir_path, self.handle_event)
        self._rpc_handle(AddDirectoryRequest(path=posixpath.relpath(sub_dir_path, self._dir_path)))
        for child in self._fs.listdir(sub_dir_path):
            full_path = posixpath.join(sub_dir_path, child)
            child_rel_path = posixpath.relpath(full_path, self._dir_path)
            if self._fs.isfile(full_path):
                self._rpc_handle(AddFileRequest(path=child_rel_path, content=self._fs.readfile(full_path)))
            else:
                self.replicate_dir_recursively(full_path)

    def handle_event(self, event: FileSystemEvent):
        """Handle a file system event.

        Used as the callback provided to FileSystem.watchdir().
        """
        # TODO
        rel_path = posixpath.relpath(event.path, self._dir_path)
        if event.event_type == FileSystemEventType.FILE_OR_SUBDIR_REMOVED:
            self._fs.unwatchdir(event.path)
            self._rpc_handle(RemoveDirectoryRequest(path=rel_path))
        elif event.event_type == FileSystemEventType.FILE_OR_SUBDIR_ADDED:
            if self._fs.isdir(event.path):
                self.replicate_dir_recursively(path=event.path)
            else:
                self._rpc_handle(AddFileRequest(path=rel_path, content=self._fs.readfile(event.path)))
            

        # I was expecting isfile to return valid value irrespective of whether the file exists or not but the assumption is invalid.
        # Did not have enough time to modify the code to remove that assumption.
        # if self._fs.isfile(event.path):
        #     if event.event_type == FileSystemEventType.FILE_OR_SUBDIR_ADDED or event.event_type == FileSystemEventType.FILE_MODIFIED:
        #         self._rpc_handle(AddFileRequest(path=rel_path, content=self._fs.readfile(event.path)))
        #     elif event.event_type == FileSystemEventType.FILE_OR_SUBDIR_REMOVED:
        #         self._rpc_handle(RemoveFileRequest(path=rel_path))
        # elif self._fs.isdir(event.path):
        #     if event.event_type == FileSystemEventType.FILE_OR_SUBDIR_ADDED:
        #         self.replicate_dir_recursively(path=event.path)
        #     elif event.event_type == FileSystemEventType.FILE_OR_SUBDIR_REMOVED:
        #         self._rpc_handle(RemoveDirectoryRequest(path=rel_path))
        #         self._fs.unwatchdir(event.path)
        
class ReplicatorTarget:
    """Class representing the target side of a file replicator."""

    def __init__(self, fs: FileSystem, dir_path: str):
        self._fs = fs
        self._dir_path = dir_path

        # TODO
        self._fs.removedir(self._dir_path)
        self._fs.makedirs(self._dir_path)

    def handle_request(self, request: Any) -> Any:
        """Handle a request from the ReplicatorSource."""
        # TODO
        full_path = posixpath.join(self._dir_path, request.path)
        self._fs.makedirs(posixpath.dirname(full_path))
        if isinstance(request, AddDirectoryRequest):
            self._fs.makedir(full_path)
        elif isinstance(request, AddFileRequest):
            self._fs.writefile(full_path, request.content)
        elif isinstance(request, RemoveFileRequest):
            self._fs.removefile(full_path)
        elif isinstance(request, RemoveDirectoryRequest):
            self._fs.removedir(full_path)
        else:
            return GenericResponse(success=False)
        return GenericResponse(success=True)